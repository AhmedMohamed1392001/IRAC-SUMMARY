"""
LLM-as-a-Judge Evaluation for IRAC Legal Summaries
Uses Claude or GPT models to evaluate model predictions against reference summaries.

Based on Human IRAC Summary Guidelines for gold-standard evaluation.
"""

import json
import os
import argparse
from pathlib import Path
from tqdm import tqdm
import anthropic
import openai
from datetime import datetime

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL CONFIGURATIONS
# ═══════════════════════════════════════════════════════════════════════════════

SUPPORTED_MODELS = {
    # Claude models
    "claude-sonnet-4-5-20250929": {"provider": "anthropic", "display_name": "Claude Sonnet 4.5"},
    # GPT models
    "gpt-5.2": {"provider": "openai", "display_name": "GPT-5.2"},
    # OpenAI o-series reasoning models
    "o3": {"provider": "openai", "display_name": "o3"},
}

def get_provider(model: str) -> str:
    """Get the provider for a given model."""
    if model in SUPPORTED_MODELS:
        return SUPPORTED_MODELS[model]["provider"]
    # Infer provider from model name
    if model.startswith("claude"):
        return "anthropic"
    elif model.startswith("gpt") or model.startswith("o1") or model.startswith("o3"):
        return "openai"
    else:
        raise ValueError(f"Unknown model: {model}. Supported models: {list(SUPPORTED_MODELS.keys())}")

# ═══════════════════════════════════════════════════════════════════════════════
# JUDGE PROMPT TEMPLATE (Based on Human IRAC Summary Guidelines)
# ═══════════════════════════════════════════════════════════════════════════════

JUDGE_PROMPT = """You are an expert legal evaluator assessing the quality of IRAC (Issue, Rule, Application, Conclusion) case summaries. Your task is to compare a MODEL OUTPUT against a REFERENCE summary written by legal experts, using the Human IRAC Summary Guidelines as the gold standard.

═══════════════════════════════════════════════════════════════════════════════
BACKGROUND: HUMAN IRAC SUMMARY GUIDELINES
═══════════════════════════════════════════════════════════════════════════════

The REFERENCE summaries were created following these principles:
• Capture the core legal problem and its resolution
• Identify the controlling rules and key authorities
• Show how the court applied those rules to the material facts
• State the bottom-line conclusion clearly and precisely

The structure follows standard IRAC/CREAC teaching materials used in legal writing programs.

═══════════════════════════════════════════════════════════════════════════════
SECTION-BY-SECTION EVALUATION CRITERIA
═══════════════════════════════════════════════════════════════════════════════

【ISSUE SECTION】
Purpose: Identify the central legal question the court resolves.

Gold Standard Requirements:
• One to two sentences only
• Framed as a precise, justiciable question of law in "whether ... when/in circumstances where ..." format
• Focus on the primary dispositive issue
• Include: relevant parties (by role: "employer", "tenant", "public authority"), the core legal doctrine (e.g. duty of care, offer and acceptance), and the key factual circumstance that makes the issue controversial

Good Example: "Whether a landlord owed a duty of care to a visitor injured in a common stairwell when the landlord had outsourced maintenance to a third-party contractor."

Avoid: Overly broad issues ("Whether the defendant was negligent") or pure factual questions ("Whether the light was broken")

Scoring:
• 2 (Correct): Identifies the correct legal question(s) with precise "whether...when" framing, includes relevant parties/roles, core doctrine, and key factual circumstance
• 1 (Partial): Main issue identified but framing is imprecise, missing parties/doctrine context, or overly broad/narrow
• 0 (Incorrect): Wrong issue identified, pure factual question, or completely misses the legal question

═══════════════════════════════════════════════════════════════════════════════

【RULE SECTION】
Purpose: State the controlling legal rule(s), test(s), or standard(s) applied to resolve the issue.

Gold Standard Requirements:
• One short paragraph or a few short paragraphs
• Start with the controlling rule used by the court, not abstract treatise
• Present rules in element or step form where possible
• Identify sources of law: statute, regulation, constitutional provision, prior cases
• State the operative legal test or elements in concise prose or numbered form
• Include any exceptions, defences, or limitations central to the court's reasoning
• Include pinpoint citations to the judgment (e.g. "[35]" or "p. 742")

Good Example: "A duty of care arises where harm is reasonably foreseeable, there is a relationship of proximity, and it is fair, just and reasonable to impose a duty (at [42]-[45])."

Do Not: List every citation; select only those necessary. Do not treat broad policy discussion as "rule" unless crystallised into a standard.

Scoring:
• 2 (Correct): Accurately states controlling rule(s) with proper sources, elements/test in correct form, relevant exceptions noted, appropriate pinpoint citations
• 1 (Partial): Main rule correct but missing elements, minor citation inaccuracies, or omits important exceptions/limitations
• 0 (Incorrect): Wrong legal rule applied, fundamental misstatement of law, or no rule stated

═══════════════════════════════════════════════════════════════════════════════

【APPLICATION / ANALYSIS SECTION】
Purpose: Show how the court applies the rule to the material facts and why it reaches its conclusion. This is the MOST IMPORTANT and should be the LONGEST section.

Gold Standard Requirements:
• Follow the structure of the rule: walk through each element or step
• Use connective language that explicitly links law and fact ("because", "since", "in light of", "given that")
• For each key element:
  - State the element in your own words
  - Identify the fact(s) the court treats as material
  - Explain the court's reasoning
  - Indicate use of precedent
  - Flag any important policy reasoning or doctrinal moves
  - Cite the judgment selectively

Priorities:
• Always cover: The central contested element(s); The factual hinge on which the court decided
• Omit: Extended recitation of arguments that didn't influence outcome; Minor factual details with no legal significance

Scoring:
• 2 (Correct): Thorough element-by-element analysis linking law to facts, captures court's reasoning faithfully, identifies contested elements and factual hinge, uses connective language, appropriate use of precedent
• 1 (Partial): Some analysis present but incomplete, tends to paraphrase or jump from rule to conclusion without clear reasoning, misses key factual applications or contested elements
• 0 (Incorrect): No meaningful analysis, fundamentally wrong application, or essentially just restates the holding without reasoning

═══════════════════════════════════════════════════════════════════════════════

【CONCLUSION SECTION】
Purpose: State the bottom-line legal outcome that answers the Issue.

Gold Standard Requirements:
• One sentence; exceptionally two if necessary
• Located at the end of the IRAC
• Explicitly answer the Issue: "The court held that [X], so [legal consequence]"
• Include: Holding on the key legal question (e.g. "no duty of care", "contract was formed")
• Include: Procedural disposition (e.g. appeal allowed/dismissed, judgment set aside, matter remitted)

Scoring:
• 2 (Correct): Correctly states both the legal holding AND procedural disposition, directly answers the Issue
• 1 (Partial): Holding correct but procedural outcome wrong/missing, OR procedural outcome correct but holding unclear
• 0 (Incorrect): Wrong holding, wrong procedural outcome, or completely misses the conclusion

═══════════════════════════════════════════════════════════════════════════════
EVALUATION PROCESS
═══════════════════════════════════════════════════════════════════════════════

Step 1: Read the REFERENCE summary completely, noting the core elements of each section
Step 2: Read the MODEL OUTPUT completely
Step 3: For ISSUE: Check framing, parties, doctrine, factual context → assign 0/1/2
Step 4: For RULE: Check rule accuracy, sources, elements, citations → assign 0/1/2
Step 5: For APPLICATION: Check element-by-element analysis, reasoning, connectives → assign 0/1/2
Step 6: For CONCLUSION: Check holding accuracy and procedural disposition → assign 0/1/2
Step 7: Return scores in JSON format

NOTE: Hallucination detection will be derived from scores:
• Score 0 in ANY section = likely contains hallucinations (fabricated content)
• Score 1 = may contain minor invented details
• Score 2 = content is grounded in reference

═══════════════════════════════════════════════════════════════════════════════
IMPORTANT GUIDELINES
═══════════════════════════════════════════════════════════════════════════════

• Focus on SEMANTIC accuracy, not exact wording - paraphrasing is acceptable if meaning preserved
• Ignore differences in writing style, grammar, or formatting
• Different section headers are acceptable (e.g., "Issue" vs "I. Issue(s)" vs "**Issue**")
• Missing information reduces score; WRONG information reduces score MORE
• When in doubt between two scores, choose the LOWER score
• The Application section should be weighted most heavily in your assessment as it is the core of legal analysis

═══════════════════════════════════════════════════════════════════════════════
REFERENCE SUMMARY:
═══════════════════════════════════════════════════════════════════════════════
{reference}

═══════════════════════════════════════════════════════════════════════════════
MODEL OUTPUT:
═══════════════════════════════════════════════════════════════════════════════
{prediction}

═══════════════════════════════════════════════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════════════════════════════════════════════

Return ONLY a valid JSON object with no additional text, markdown, or explanation:
{{"issue_score": <0|1|2>, "rule_score": <0|1|2>, "application_score": <0|1|2>, "conclusion_score": <0|1|2>}}"""


# Hallucination detection threshold
# Based on IslamicLegalBench methodology:
# - Score 0 = Incorrect → automatically flagged as hallucination
# - Score 1 = Partial → may contain invented details (use threshold)
# - Score 2 = Correct → no hallucination
HALLUCINATION_THRESHOLD = 1.0  # Average score below this = hallucination


def derive_hallucination(scores: dict, threshold: float = 1.0) -> dict:
    """
    Derive hallucination flag from section scores.

    Based on IslamicLegalBench methodology:
    - Any section with score 0 (Incorrect) → hallucination = True
    - Average score below threshold with any score 1 → hallucination = True
    - All scores >= 2 → hallucination = False

    Args:
        scores: dict with issue_score, rule_score, application_score, conclusion_score
        threshold: average score threshold for hallucination detection

    Returns:
        dict with hallucination flag and detection method
    """
    section_scores = [
        scores.get("issue_score", -1),
        scores.get("rule_score", -1),
        scores.get("application_score", -1),
        scores.get("conclusion_score", -1)
    ]

    # Filter out error scores (-1)
    valid_scores = [s for s in section_scores if s >= 0]

    if not valid_scores:
        return {"hallucination": None, "detection_method": "error"}

    min_score = min(valid_scores)
    avg_score = sum(valid_scores) / len(valid_scores)

    # Rule 1: Any score of 0 (Incorrect) → automatic hallucination
    if min_score == 0:
        return {
            "hallucination": True,
            "detection_method": "score_0_detected",
            "min_score": min_score,
            "avg_score": round(avg_score, 2)
        }

    # Rule 2: Average below threshold with any partial score → hallucination
    if avg_score < threshold and min_score == 1:
        return {
            "hallucination": True,
            "detection_method": "below_threshold_with_partial",
            "min_score": min_score,
            "avg_score": round(avg_score, 2)
        }

    # Rule 3: All scores >= 1 and average >= threshold → no hallucination
    return {
        "hallucination": False,
        "detection_method": "scores_acceptable",
        "min_score": min_score,
        "avg_score": round(avg_score, 2)
    }


def evaluate_single(client, reference: str, prediction: str, model: str = "claude-sonnet-4-5-20250929", hallucination_threshold: float = 1.0, provider: str = "anthropic") -> dict:
    """
    Evaluate a single prediction against reference using Claude or GPT.
    """
    prompt = JUDGE_PROMPT.format(reference=reference, prediction=prediction)

    try:
        if provider == "anthropic":
            message = client.messages.create(
                model=model,
                max_tokens=200,
                temperature=0.0,  # Deterministic output
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            response_text = message.content[0].text.strip()
        elif provider == "openai":
            # o-series models (o1, o3) don't support temperature parameter
            is_o_series = model.startswith("o1") or model.startswith("o3")

            if is_o_series:
                response = client.chat.completions.create(
                    model=model,
                    max_completion_tokens=2000,  # o-series needs more tokens for reasoning
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
            else:
                response = client.chat.completions.create(
                    model=model,
                    max_completion_tokens=200,
                    temperature=0.0,  # Deterministic output
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )

            # Handle potential None response
            if response.choices[0].message.content is None:
                raise ValueError(f"Model returned empty response. Finish reason: {response.choices[0].finish_reason}")
            response_text = response.choices[0].message.content.strip()
        else:
            raise ValueError(f"Unknown provider: {provider}")

        # Parse the response

        # Try to extract JSON from response
        # Handle cases where model might add extra text
        if response_text.startswith("{"):
            json_str = response_text
        else:
            # Try to find JSON in response
            import re
            json_match = re.search(r'\{[^{}]+\}', response_text)
            if json_match:
                json_str = json_match.group()
            else:
                raise ValueError(f"Could not parse JSON from response: {response_text[:500] if response_text else 'EMPTY'}")

        result = json.loads(json_str)

        # Validate the result (hallucination is now derived from scores)
        required_keys = ["issue_score", "rule_score", "application_score", "conclusion_score"]
        for key in required_keys:
            if key not in result:
                raise ValueError(f"Missing key in response: {key}")

        # Derive hallucination from scores
        hallucination_result = derive_hallucination(result, threshold=hallucination_threshold)
        result["hallucination"] = hallucination_result["hallucination"]
        result["hallucination_method"] = hallucination_result["detection_method"]

        return result

    except Exception as e:
        print(f"Error evaluating: {e}")
        # Return default scores on error
        return {
            "issue_score": -1,
            "rule_score": -1,
            "application_score": -1,
            "conclusion_score": -1,
            "hallucination": None,
            "hallucination_method": "error",
            "error": str(e)
        }


def run_evaluation(predictions_path: str, output_dir: str = None, model: str = "claude-sonnet-4-5-20250929", hallucination_threshold: float = 1.0):
    """
    Run LLM-as-a-Judge evaluation on all predictions.
    """
    # Determine provider
    provider = get_provider(model)

    # Check for API key and initialize client
    if provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        client = anthropic.Anthropic(api_key=api_key)
    elif provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        client = openai.OpenAI(api_key=api_key)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    # Load predictions
    print(f"Loading predictions from {predictions_path}...")
    with open(predictions_path, 'r') as f:
        predictions = json.load(f)

    print(f"Found {len(predictions)} samples to evaluate")
    print(f"Using model: {model} (provider: {provider})")
    print(f"Using Human IRAC Summary Guidelines as evaluation criteria")
    print()

    # Run evaluation
    results = []

    for item in tqdm(predictions, desc="Evaluating"):
        eval_result = evaluate_single(
            client=client,
            reference=item["reference"],
            prediction=item["prediction"],
            model=model,
            hallucination_threshold=hallucination_threshold,
            provider=provider
        )

        results.append({
            "index": item["index"],
            "case_name": item["case_name"],
            "issue_score": eval_result["issue_score"],
            "rule_score": eval_result["rule_score"],
            "application_score": eval_result["application_score"],
            "conclusion_score": eval_result["conclusion_score"],
            "hallucination": eval_result["hallucination"],
            "hallucination_method": eval_result.get("hallucination_method", None),
            "error": eval_result.get("error", None)
        })

    # Calculate aggregate metrics
    valid_results = [r for r in results if r["issue_score"] >= 0]

    if valid_results:
        avg_issue = sum(r["issue_score"] for r in valid_results) / len(valid_results)
        avg_rule = sum(r["rule_score"] for r in valid_results) / len(valid_results)
        avg_application = sum(r["application_score"] for r in valid_results) / len(valid_results)
        avg_conclusion = sum(r["conclusion_score"] for r in valid_results) / len(valid_results)
        avg_overall = (avg_issue + avg_rule + avg_application + avg_conclusion) / 4
        hallucination_rate = sum(1 for r in valid_results if r["hallucination"]) / len(valid_results)

        # Normalize to 0-100 scale
        avg_issue_pct = (avg_issue / 2) * 100
        avg_rule_pct = (avg_rule / 2) * 100
        avg_application_pct = (avg_application / 2) * 100
        avg_conclusion_pct = (avg_conclusion / 2) * 100
        avg_overall_pct = (avg_overall / 2) * 100
    else:
        avg_issue = avg_rule = avg_application = avg_conclusion = avg_overall = 0
        avg_issue_pct = avg_rule_pct = avg_application_pct = avg_conclusion_pct = avg_overall_pct = 0
        hallucination_rate = 0

    # Count hallucination detection methods
    hallucination_methods = {}
    for r in valid_results:
        method = r.get("hallucination_method", "unknown")
        hallucination_methods[method] = hallucination_methods.get(method, 0) + 1

    # Prepare summary
    summary = {
        "evaluation_date": datetime.now().isoformat(),
        "model_used": model,
        "evaluation_criteria": "Human IRAC Summary Guidelines",
        "hallucination_detection": {
            "method": "score-based (IslamicLegalBench methodology)",
            "threshold": hallucination_threshold,
            "rules": [
                "Score 0 in ANY section → hallucination = True",
                f"Average < {hallucination_threshold} with partial scores → hallucination = True",
                "All scores acceptable → hallucination = False"
            ],
            "detection_breakdown": hallucination_methods
        },
        "total_samples": len(predictions),
        "valid_evaluations": len(valid_results),
        "errors": len(predictions) - len(valid_results),
        "metrics": {
            "issue_score_avg": round(avg_issue, 3),
            "rule_score_avg": round(avg_rule, 3),
            "application_score_avg": round(avg_application, 3),
            "conclusion_score_avg": round(avg_conclusion, 3),
            "overall_score_avg": round(avg_overall, 3),
            "issue_score_pct": round(avg_issue_pct, 1),
            "rule_score_pct": round(avg_rule_pct, 1),
            "application_score_pct": round(avg_application_pct, 1),
            "conclusion_score_pct": round(avg_conclusion_pct, 1),
            "overall_score_pct": round(avg_overall_pct, 1),
            "hallucination_rate": round(hallucination_rate, 3),
            "hallucination_rate_pct": round(hallucination_rate * 100, 1)
        },
        "results": results
    }

    # Determine output path
    if output_dir is None:
        output_dir = Path(predictions_path).parent
    else:
        output_dir = Path(output_dir)

    output_path = output_dir / f"llm_judge_results_{model}.json"

    # Save results
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 60)
    print("EVALUATION COMPLETE (Human IRAC Summary Guidelines)")
    print("=" * 60)
    print()
    print(f"Results saved to: {output_path}")
    print()
    print("SECTION SCORES (0-2 scale, higher is better):")
    print(f"  Issue:        {avg_issue:.2f} / 2.00  ({avg_issue_pct:.1f}%)")
    print(f"  Rule:         {avg_rule:.2f} / 2.00  ({avg_rule_pct:.1f}%)")
    print(f"  Application:  {avg_application:.2f} / 2.00  ({avg_application_pct:.1f}%)")
    print(f"  Conclusion:   {avg_conclusion:.2f} / 2.00  ({avg_conclusion_pct:.1f}%)")
    print()
    print(f"OVERALL SCORE:  {avg_overall:.2f} / 2.00  ({avg_overall_pct:.1f}%)")
    print()
    print(f"HALLUCINATION DETECTION (threshold={hallucination_threshold}):")
    print(f"  Rate: {hallucination_rate*100:.1f}%")
    print(f"  Method breakdown: {hallucination_methods}")
    print()

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM-as-a-Judge Evaluation for IRAC Summaries")
    parser.add_argument(
        "--predictions",
        type=str,
        default="models_output/Qwen4B/predictions_simple.json",
        help="Path to predictions JSON file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save results (default: same as predictions)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-sonnet-4-5-20250929",
        help="Model to use. Claude: claude-sonnet-4-5-20250929. GPT: gpt-5.2, gpt-4o. OpenAI o-series: o3, o3-mini, o1, o1-mini (default: claude-sonnet-4-5-20250929)"
    )
    parser.add_argument(
        "--hallucination-threshold",
        type=float,
        default=1.0,
        help="Average score threshold for hallucination detection (default: 1.0)"
    )

    args = parser.parse_args()

    run_evaluation(
        predictions_path=args.predictions,
        output_dir=args.output_dir,
        model=args.model,
        hallucination_threshold=args.hallucination_threshold
    )

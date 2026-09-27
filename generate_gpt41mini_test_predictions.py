"""Generate GPT-4.1-mini (fine-tuned) predictions for the 31 held-out test cases.

Usage:
  $env:OPENAI_API_KEY="sk-..."
  py generate_gpt41mini_test_predictions.py

Then score with the existing judge script:
  py llm_judge_eval.py --predictions predictions_for_judge_gpt41mini_TEST.json --model o3 --output-dir results_gpt41mini_test
"""
import json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "ft:gpt-4.1-mini-2025-04-14:personal:irac-summary:Cy4JZaLj"
TEMPERATURE = 0.7
MAX_TOKENS = 1024

# --- reconstruct the exact prompt template from the training file ---
with open(os.path.join(HERE, "train_gpt41_irac_instruction_format.jsonl"), encoding="utf8") as f:
    rec = json.loads(f.readline())
user = rec["messages"][0]["content"]
q_idx = user.index("### Question:")
cn_idx = user.index("[Case Name] ", q_idx)
PREFIX = user[:cn_idx] + "[Case Name] "
SUFFIX = "Output:\n"
assert user.endswith(SUFFIX)
print("Template reconstructed; prefix ends:", repr(PREFIX[-30:]))

def build_prompt(name, text):
    body = text if text.endswith("\n\n") else text.rstrip("\n") + "\n\n"
    return PREFIX + name + "\n[Summary] " + body + SUFFIX

with open(os.path.join(HERE, "test_set_31_for_gpt41mini.json"), encoding="utf8") as f:
    cases = json.load(f)
print(f"{len(cases)} test cases loaded")

try:
    from openai import OpenAI
except ImportError:
    sys.exit("Run first:  py -m pip install openai")
client = OpenAI()

out_path = os.path.join(HERE, "predictions_for_judge_gpt41mini_TEST.json")
results = []
for i, c in enumerate(cases):
    prompt = build_prompt(c["case_name"], c["source_text"])
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            pred = resp.choices[0].message.content
            break
        except Exception as e:
            print(f"  attempt {attempt+1} failed for case {i}: {e}")
            time.sleep(10)
    else:
        pred = ""
        print(f"  !! case {i} FAILED after 3 attempts")
    results.append({
        "index": i,
        "case_number": i,
        "case_name": c["case_name"],
        "reference": c["reference"],
        "prediction": pred,
        "source_text": c["source_text"],
    })
    print(f"[{i+1}/31] {c['case_name'][:55]}  -> {len(pred)} chars")
    with open(out_path, "w", encoding="utf8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)

empty = [r["case_name"] for r in results if not r["prediction"].strip()]
print("\nDone. Wrote", out_path)
print("Empty predictions:", empty if empty else "none")

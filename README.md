# IRAC-Bench: Benchmarking Legal Reasoning Fidelity in Judicial Summarization

Code and dataset for the paper *IRAC-Bench: Benchmarking Legal Reasoning Fidelity in Judicial Summarization*.

## Overview

IRAC-Bench is a benchmark of **299 common-law judicial decisions** from courts in the United Kingdom and the United States, each paired with a human-validated **Issue, Rule, Application, Conclusion (IRAC)** reference annotation. The benchmark treats judicial summarization as structured legal reasoning extraction rather than generic text compression, and scores Rule and Application separately as distinct reasoning operations.

## What is in this repository

- **IRAC reference annotations** for all 299 decisions (CC BY 4.0)
- **Dataset splits**: 237 training / 31 development / 31 test cases (stratified by jurisdiction and legal domain, seed 42)
- **Per-case provenance manifest**: court citations, decision dates, and retrieval dates
- **Human IRAC Summary Guidelines**: the full instructions given to the legally trained annotators
- **Evaluation code**: the LLM-as-judge evaluation script (`llm_judge_eval.py`)

## Annotation workflow

Candidate IRAC summaries were drafted with model assistance using controlled prompts, then each draft was checked against the full judicial opinion and substantively revised by a legally trained annotator (a Professor of Law or a fourth-year law student with formal IRAC training). No unreviewed model output was retained. Because each case was validated by a single annotator rather than independently double-annotated, the annotations are described as human-validated reference annotations.

## Evaluation

Generated summaries are scored per IRAC component on a 0-2 ordinal scale by an LLM judge (OpenAI o3, temperature 0), validated against a human evaluation by three legally trained annotators. The **reliability-failure rate** is the proportion of summaries with at least one component scored 0 or a mean component score below 1.0.

### Headline results (31-case held-out test set)

| Model | Overall IRAC | Reliability-failure rate |
| --- | --- | --- |
| Claude Sonnet 4.5 (zero-shot) | 87.1% | 3.2% |
| Gemini 3 Flash (zero-shot) | 87.1% | 9.7% |
| GPT-4.1-mini (fine-tuned) | 87.1% | 4.4% |
| Gemini 3 Pro (zero-shot) | 85.1% | 9.7% |
| GPT-5.2 (zero-shot) | 81.9% | 12.9% |
| DeepSeek-V3.2 (zero-shot) | 80.6% | 9.7% |
| Grok 4.1 (zero-shot) | 75.0% | 26.7% |
| Qwen3-14B-11000 (fine-tuned) | 71.2% | 6.7% |
| Phi-4-11000 (fine-tuned) | 68.5% | 16.1% |
| Qwen3-235B (zero-shot) | 63.7% | 74.2% |
| Qwen3-4B-11000 (fine-tuned) | 59.7% | 41.9% |
| Mistral-7B-11000 (fine-tuned) | 53.2% | 48.4% |
| Llama-3.2-11000 (fine-tuned) | 51.6% | 48.4% |
| Qwen3-4B-8194 (fine-tuned) | 50.4% | 54.8% |
| Mistral-7B-12000 (fine-tuned) | 43.1% | 64.5% |

Surface metrics do not certify legal reliability: Qwen3-235B reaches ROUGE-1 61.61 and BERTScore F1 85.94 while failing on 74.2% of test summaries.

## Usage

```bash
pip install pandas scikit-learn anthropic openai tqdm

# evaluate model predictions with the LLM-as-judge framework
python llm_judge_eval.py --predictions models_output/predictions.json
```

Set `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` as needed.

## Licensing

- **Code**: MIT License (see `LICENSE`)
- **IRAC annotations, splits, and provenance manifest**: CC BY 4.0 (see `DATA_LICENSE`)
- **Judicial opinions**: public-domain texts from open-access legal repositories and official court websites; the provenance manifest supports verification of reuse conditions per case.

## Citation

If you use IRAC-Bench, please cite the paper *IRAC-Bench: Benchmarking Legal Reasoning Fidelity in Judicial Summarization* (see `CITATION.cff`).

## Acknowledgments

The authors thank Qatar University and Bond University for institutional support, and the legal experts who participated in the IRAC annotation review process.

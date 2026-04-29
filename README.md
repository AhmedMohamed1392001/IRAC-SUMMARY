# IRAC Legal Case Document Summarization

## Overview

This project implements **structured legal reasoning extraction** for judicial case decisions using the **IRAC framework** (Issue, Rule, Application, Conclusion). Unlike generic text compression, this approach focuses on extracting the core reasoning components that legal professionals use when briefing and communicating cases.

The system evaluates **large language models** on their ability to produce professionally usable legal summaries grounded in legal faithfulness and reasoning fidelity, rather than surface-level lexical similarity.

## Key Features

- **IRAC-Based Structured Summarization**: Decomposes judicial decisions into Issue, Rule, Application, and Conclusion components
- **LLM-as-Judge Evaluation**: Employs Claude or GPT models as expert evaluators using Human IRAC Summary Guidelines
- **Hallucination Detection**: Identifies and flags legally consequential errors such as missing rules, distorted reasoning, and incorrect outcomes
- **Multi-Model Evaluation**: Tests performance across frontier LLMs (Claude, GPT, Gemini, DeepSeek, Qwen, Mistral, etc.)
- **Legal Faithfulness Metrics**: Prioritizes structural fidelity and reasoning accuracy over lexical overlap
- **Component-Level Scoring**: Enables fine-grained assessment of each IRAC section independently

## Domain

**Legal AI & Natural Language Processing** – Specifically designed for:
- Legal professionals requiring reliable case summaries
- Researchers studying legal document understanding
- Legal tech systems needing trustworthy case briefing
- Domain experts in judicial reasoning extraction

## Project Structure

```
├── llm_judge_eval.py          # LLM-as-Judge evaluation script
├── prepare_data.py             # Data preparation and train/dev/test splitting
├── data/
│   ├── raw/
│   │   └── DataCases.xlsx     # Input legal case data
│   └── processed/
│       ├── train.jsonl         # Training dataset
│       ├── dev.jsonl           # Development/validation dataset
│       └── test.jsonl          # Test dataset
└── README.md                    # This file
```

## Installation & Setup

### Prerequisites
- Python 3.9+
- Required libraries: `pandas`, `scikit-learn`, `anthropic`, `openai`, `tqdm`

### Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd irac-legal-summarization
   ```

2. **Install dependencies**
   ```bash
   pip install pandas scikit-learn anthropic openai tqdm
   ```

3. **Set up API credentials**
   - For Claude: `export ANTHROPIC_API_KEY="your-key"`
   - For GPT: `export OPENAI_API_KEY="your-key"`

## Usage

### 1. Data Preparation

Prepare your legal case data from an XLSX file:

```bash
python prepare_data.py
```

This script:
- Reads `data/raw/DataCases.xlsx`
- Splits data into train (80%), dev (10%), and test (10%) sets
- Outputs JSONL files to `data/processed/`
- Handles missing data gracefully with placeholders

### 2. Evaluate Model Predictions

Evaluate any model's predictions using the LLM-as-Judge framework:

```bash
# Evaluate with Claude Sonnet 4.5 (default)
python llm_judge_eval.py --predictions models_output/predictions.json

# Evaluate with GPT-5.2
python llm_judge_eval.py --predictions models_output/predictions.json --model gpt-5.2

# Custom output directory
python llm_judge_eval.py \
    --predictions models_output/predictions.json \
    --output-dir results/ \
    --model claude-sonnet-4-5-20250929 \
    --hallucination-threshold 1.0
```

### 3. Supported Models

**Zero-Shot Evaluation:**
- Claude Sonnet 4.5 (Anthropic)
- GPT-5.2 (OpenAI)
- Gemini 3 Flash / Gemini 3 Pro (DeepMind)
- DeepSeek v3.2
- Grok 4.1 (xAI)
- Qwen-235B

**Fine-Tuned Models:**
- Qwen3-14B
- Mistral-7B
- Llama3.2
- Phi-4

## IRAC Framework Explained

Each case summary is structured into four legally meaningful components:

### Issue
- **Purpose**: Identify the central legal question
- **Format**: "Whether [X] when [circumstances]"
- **Scoring**: Precise framing, includes parties, doctrine, and key facts (0-2 scale)

**Example**: "Whether a landlord owed a duty of care to a visitor injured in a common stairwell when the landlord had outsourced maintenance to a third-party contractor."

### Rule
- **Purpose**: State the controlling legal rules, tests, or standards
- **Format**: Elements/steps with proper sources and pinpoint citations
- **Scoring**: Accuracy, completeness of elements, relevant exceptions (0-2 scale)

**Example**: "A duty of care arises where harm is reasonably foreseeable, there is a relationship of proximity, and it is fair, just and reasonable to impose a duty (at [42]-[45])."

### Application
- **Purpose**: Show how the court applies the rule to material facts
- **Format**: Element-by-element analysis with explicit law-to-fact connections
- **Scoring**: Thoroughness, legal reasoning quality, identification of contested elements (0-2 scale)
- **Note**: This is the most critical section and typically the longest

### Conclusion
- **Purpose**: State the bottom-line legal outcome
- **Format**: One sentence answering the Issue + procedural disposition
- **Scoring**: Correctness of both holding and procedural outcome (0-2 scale)

**Example**: "The court held that the landlord owed a duty of care; therefore, the appeal was allowed."

## Evaluation Metrics

### Primary Metrics (Structure-Based)

| Metric | Description | Scale |
|--------|-------------|-------|
| Issue Score | Legal question accuracy | 0-2 |
| Rule Score | Controlling rule identification | 0-2 |
| Application Score | Reasoning quality and completeness | 0-2 |
| Conclusion Score | Holding and disposition accuracy | 0-2 |
| **Overall Score** | **Mean across all sections** | **0-2** |

### Secondary Metrics (Lexical/Semantic)

- **ROUGE-1, ROUGE-2, ROUGE-L**: N-gram overlap (diagnostic only)
- **BERTScore F1**: Semantic similarity using contextual embeddings

### Hallucination Detection

Models are flagged as hallucinating if:
1. Any section receives a score of 0 (missing, incorrect, or fabricated), OR
2. Average score falls below 1.0 with partial scores

**Hallucination Rate** = % of test samples with hallucinations

## Benchmark Results

Key findings from IRAC-BENCH evaluation (300 judicial decisions):

| Model | Overall Score | Hallucination Rate | Best At | Notes |
|-------|---------------|--------------------|---------|-------|
| Claude Sonnet 4.5 | 87.1% | 3.2% | Application, Conclusion | Lowest hallucination |
| Gemini 3 Flash | 87.1% | 9.7% | Rule extraction | Close lexical alignment |
| GPT-4.1-mini (FT) | 87.1% | 4.4% | All components | Fine-tuning effective |
| DeepSeek v3.2 | 80.6% | 9.7% | Overall balance | Competitive zero-shot |
| Mistral-7B (FT) | 53.2% | 48.4% | Smaller capacity | Struggles with analysis |
| Qwen-235B | 63.7% | 74.2% | N/A | High hallucination risk |

**Key Insight**: Surface metrics (ROUGE, BERTScore) are insufficient. Qwen-235B achieved ROUGE-1 of 61.61 but had 74.2% hallucination rate.

## Output Format

The evaluation produces a JSON file with:

```json
{
  "evaluation_date": "2025-04-29T...",
  "model_used": "claude-sonnet-4-5-20250929",
  "evaluation_criteria": "Human IRAC Summary Guidelines",
  "metrics": {
    "issue_score_pct": 88.7,
    "rule_score_pct": 75.8,
    "application_score_pct": 90.3,
    "conclusion_score_pct": 93.5,
    "overall_score_pct": 87.1,
    "hallucination_rate_pct": 3.2
  },
  "results": [
    {
      "case_name": "Case v. Party",
      "issue_score": 2,
      "rule_score": 1,
      "application_score": 2,
      "conclusion_score": 2,
      "hallucination": false
    }
  ]
}
```

## Research Methodology

This project implements research advancing **structured legal reasoning extraction** rather than generic compression:

1. **Dataset**: 300 curated judicial decisions from multiple jurisdictions (UK Supreme Court, US Supreme Court, federal/state courts)
2. **Annotation Pipeline**: LLM-generated candidates + expert human validation (substantive revision)
3. **Evaluation**: LLM-as-Judge (using o3) anchored in legal reasoning rubric
4. **Validation**: Human expert evaluation on representative sample confirms automated scoring reliability

## Requirements & Dependencies

```
pandas>=1.3.0
scikit-learn>=1.0.0
anthropic>=0.8.0
openai>=1.0.0
tqdm>=4.62.0
```

Install all:
```bash
pip install -r requirements.txt
```

## Legal Faithfulness vs. Surface Similarity

This project prioritizes **legal faithfulness** (does the summary preserve correct legal reasoning?) over **surface similarity** (do the words match?).

### Why This Matters

A summary can achieve high ROUGE scores while being legally wrong:
- Missing the controlling rule
- Misapplying the rule to facts
- Stating the wrong outcome
- Hallucinating doctrinal elements

**Example**: ROUGE-1=61.61, BERTScore=85.94, but 74.2% hallucination rate = **unusable in legal practice**

## Error Analysis

Common failure modes identified:

| Component | Challenge | Why It Matters |
|-----------|-----------|----------------|
| **Rule** | Identifying correct legal standard | Misstated rules change case meaning |
| **Application** | Linking law to facts element-by-element | Wrong analysis undermines holding |
| **Conclusion** | Stating both holding AND disposition | Omitted procedural outcome misleads readers |

**Key Finding**: Application (rule-to-fact linking) is the most challenging for smaller models.

## Limitations & Future Work

### Current Limitations
- Focused on English-language common law decisions
- Dataset skew toward civil procedure and tort law
- Evaluation limited to English-language jurisdictions

### Future Directions
- Multi-case reasoning and cross-citation analysis
- Retrieval-augmented generation (RAG) for better rule identification
- Cross-jurisdictional adaptation
- Improved handling of complex multi-party disputes

## Citation

If you use this project in research, please cite:

Toward Summarizing Case Decisions via Extracting Argument Issues, Rule, Analysis and
Conclusions

## License

MIT License

## Contact & Support

For questions or issues:
- **Email**: am2401295@qu.edu.qa

- **Documentation**: See paper : Toward Summarizing Case Decisions via Extracting Argument Issues, Rule, Analysis and
Conclusions

## Acknowledgments

- Legal expert annotators for IRAC-BENCH validation
- Qatar University & STM Document Engineering for dataset curation
- Anthropic, OpenAI, and other model providers for API access

---

**Last Updated**: April 2025  
**Status**: Active research project
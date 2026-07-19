# Multilingual LLM Calibration Across Turkic and Slavic Languages

Are large language models as well-calibrated in low-resource languages as they
are in English? This project measures whether models that say *"I'm 90% sure"*
are actually right about 90% of the time, and whether both accuracy and that
honesty hold up across a resource spectrum spanning two language families:
**Turkic (Uzbek, Kazakh)** and **Slavic (Russian, Macedonian)**, anchored by
**English** as the high-resource reference point.

We do **not** train any models. We test three frontier API models (GPT-4o,
Gemini, Claude) on the same 500 parallel questions across all five languages,
and compare their accuracy and calibration, with paired bootstrap significance
testing and multiple-comparisons correction.

## Research question

Does LLM accuracy and calibration degrade in lower-resource languages, and
does that degradation generalize across unrelated language families, or is it
specific to one? Testing a single low-resource language cannot distinguish a
general resource-scarcity effect from a language-specific quirk, so this
study spans two independent low-resource/high-resource pairs: Uzbek/Kazakh
(Turkic) and Russian/Macedonian (Slavic).

We measure this on the **Belebele** benchmark, a parallel multiple-choice
reading-comprehension dataset, using verbalized (0-100) confidence and
several calibration metrics, with statistical tests for whether observed
differences are distinguishable from sampling noise.

## What the pipeline does

For every question, in every language, for every model:

1. show the model the passage, the question and four options
2. get its answer and its stated confidence (0-100)
3. record the answer, the confidence and the correct answer
4. mark it right or wrong
5. measure calibration: does confidence match accuracy?
6. test whether differences between languages are statistically significant,
   correcting for the number of comparisons made

## Project structure

```
llm-calibration-low-resource-languages/
├── README.md                     this file
├── LICENSE                       MIT license
├── requirements.txt              Python packages to install
├── .gitignore                    tells git NOT to upload keys, large outputs
├── .env.example                  template showing which API keys are needed
├── config/
│   └── config.yaml               settings: models, languages, sample size, paths
├── src/
│   ├── __init__.py
│   ├── load_data.py              downloads Belebele, builds the fixed 500-question
│   │                             sample, aligned across all five languages   (run 1st)
│   ├── add_language.py           merges an additional language's content onto the
│   │                             fixed sample via the (link, question_number) key,
│   │                             used to assemble the Kazakh and Macedonian columns
│   ├── prompts.py                the prompt template
│   ├── models.py                 one wrapper per model / API
│   ├── parse.py                  pulls the answer + confidence out of a reply
│   ├── run_eval.py               the main loop: ask, record, score. Supports
│   │                             --languages to run a subset without re-querying
│   │                             (and re-paying for) languages already evaluated (run 2nd)
│   ├── compute_metrics.py        accuracy, ECE, Brier score, bootstrap CIs, and
│   │                             PAIRED bootstrap significance tests between
│   │                             every language pair, per model                (run 3rd)
│   ├── apply_fdr_correction.py   adds Benjamini-Hochberg FDR-corrected p-values
│   │                             to the pairwise significance table            (run 4th)
│   └── visualize.py              reliability diagrams, confidence histograms,
│                                  accuracy/ECE comparison charts, ECE heatmap
│                                  — all generated from metrics.csv              (run 5th)
├── data/
│   └── sample/                   the fixed 500-question sample, all 5 languages
├── results/
│   ├── raw_outputs/              every model's raw predictions, one CSV per model
│   ├── processed/                metrics.csv and pairwise_significance.csv
│   └── figures/                  all figures, saved as both .png and .pdf
├── notebooks/
│   └── analysis.ipynb            exploratory validation of dataset alignment
└── paper/                        research protocol notes + LaTeX project
```

## Language selection

Five languages across two families, anchored by a shared high-resource
reference:

| Family | High-resource | Low-resource |
|---|---|---|
| — | English | — |
| Slavic | Russian | Macedonian |
| Turkic | — | Uzbek, Kazakh |

Macedonian is the lowest-resource Slavic language available in Belebele.
Testing two independent low-resource languages from two unrelated families,
against two high-resource points, allows the study to ask whether any
observed resource-scarcity effect is general or family-specific, rather than
resting the claim on a single language pair.

## Evaluation Metrics

- Accuracy (+ 95% bootstrap confidence intervals)
- Expected Calibration Error, ECE (+ 95% bootstrap confidence intervals)
- Brier Score
- Reliability diagrams
- Paired bootstrap significance tests between every language pair, per model,
  for both accuracy and ECE (1,000 resamples) — "paired" because every
  language is evaluated on the same 500 underlying questions, which removes
  per-question difficulty as a source of noise
- Benjamini-Hochberg FDR correction across all pairwise tests

## Setup

1. Clone the repository and enter it:
   ```
   git clone https://github.com/dianakhajieva/llm-calibration-low-resource-languages
   cd llm-calibration-low-resource-languages
   ```

2. Create a virtual environment and install dependencies:
   ```
   python -m venv .venv
   source .venv/bin/activate         # on Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Add your API keys:
   ```
   cp .env.example .env
   ```
   then open `.env` and paste in the keys for the providers you use (OpenAI,
   Google, Anthropic).

## How to run

Run everything **from the project root**.

**Step 1 — build the fixed sample:**
```
python -m src.load_data
python -m src.add_language --code kaz_Cyrl --prefix kk
python -m src.add_language --code mkd_Cyrl --prefix mk
```
This produces one fixed 500-question sample with all five languages aligned
question-for-question.

**Step 2 — run the models:**
```
python -m src.run_eval
```
To run a subset of languages without re-querying ones you already have
results for:
```
python -m src.run_eval --languages kaz_Cyrl
```

**Step 3 — compute metrics and significance tests:**
```
python -m src.compute_metrics
python -m src.apply_fdr_correction
```

**Step 4 — generate figures:**
```
python -m src.visualize
```

TIP: while testing changes, set `sample_size` to a small number (like 10) in
`config/config.yaml` so runs are fast and cheap. The real results use 500.

## Choosing what to run

Edit `config/config.yaml` to change the languages, the sample size, and the
list of models. You should not need to edit the scripts themselves to change
these.

## Confidence method

This project uses **verbalized confidence**: the model is asked to state a
0-100 confidence number alongside its answer. A logprob-based confidence
method was considered but not implemented in the final results, since none of
the three evaluated models (GPT-4o, Gemini, Claude) expose it in a directly
comparable way in this pipeline; this is noted as a direction for future work.

## Reproducibility

The random seed in `config/config.yaml` fixes which questions are sampled, so
the study reproduces exactly. `add_language.py` merges each additional
language onto the same fixed 500 questions rather than re-sampling, so every
language is evaluated on identical content. Raw model replies are saved in
`results/raw_outputs/` so metrics can be recomputed without paying for the API
calls again. All bootstrap procedures use a fixed seed for reproducibility.

## Data and credit

Belebele: Bandarkar et al., *The Belebele Benchmark: a Parallel Reading
Comprehension Dataset in 122 Language Variants* (2024). Loaded from the
`facebook/belebele` dataset on Hugging Face.
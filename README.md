# Multilingual LLM Calibration

Are large language models as well-calibrated in low-resource languages as they
are in English? This project measures whether models that say *"I'm 90% sure"*
are actually right about 90% of the time, and whether that honesty holds up
across **English, Russian and Uzbek**.

We do **not** train any models. We test existing ones (API models like GPT and
Gemini, and open-weight models like Qwen, Llama and Aya) on the same parallel
questions in three languages, and compare their calibration.

## Research question

When a model answers in Uzbek or Russian rather than English, does its stated
(or internal) confidence still match how often it is actually correct, or does
it become overconfident? We measure this on the **Belebele** benchmark, a
parallel multiple-choice reading-comprehension dataset, using two ways of
reading a model's confidence and several calibration metrics.

## What the pipeline does

For every question, in every language, for every model:

1. show the model the passage, the question and four options
2. get its answer, and its confidence (either a stated 0-100 number, or its
   internal probability for the chosen option)
3. record the answer, the confidence and the correct answer
4. mark it right or wrong
5. measure calibration: does confidence match accuracy?

## Project structure

```
multilingual-llm-calibration/
├── README.md              this file
├── LICENSE                MIT license (edit names before publishing)
├── requirements.txt       Python packages to install
├── .gitignore             tells git NOT to upload keys, large outputs
├── .env.example           template showing which API keys are needed
├── config/
│   └── config.yaml        your settings: models, languages, sample size, paths
├── src/
│   ├── utils.py           finds the project root, loads config
│   ├── load_data.py       downloads Belebele, makes the fixed sample   (run 1st)
│   ├── prompts.py         the two prompt templates
│   ├── models.py          one wrapper per model / API
│   ├── parse.py           pulls the answer + confidence out of a reply
│   ├── run_eval.py        the main loop: ask, record, score            (run 2nd)
│   └── metrics.py         accuracy, calibration error, reliability diagrams
├── data/
│   ├── raw/               (empty; scratch space, git-ignored)
│   └── sample/            the fixed question sample lands here
├── results/
│   ├── raw_outputs/       every raw model reply (git-ignored, can be large)
│   └── processed/         results.csv, the one clean results table
├── notebooks/
│   └── analysis.ipynb     load results, make the metrics table + plots  (run 3rd)
├── figures/               final reliability diagrams for the paper
└── paper/                 your LaTeX / Overleaf project
```

## Setup

1. Clone the repository and enter it:
   ```
   git clone <your-repo-url>
   cd multilingual-llm-calibration
   ```

2. Create a virtual environment and install dependencies:
   ```
   python -m venv .venv
   source .venv/bin/activate         # on Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
   If you will run the open-weight models, install PyTorch for your machine
   first, following https://pytorch.org/get-started/locally/ .

3. Add your API keys:
   ```
   cp .env.example .env
   ```
   then open `.env` and paste in the keys for the providers you use.

## How to run

Run everything **from the project root**.

**Step 1 — download the data and create the fixed sample:**
```
python src/load_data.py
```
This creates `data/sample/belebele_sample.jsonl`. Open it and read a few of the
Uzbek questions to confirm they look correct (your native-speaker check).

**Step 2 — run the experiment:**
```
python src/run_eval.py
```
This fills `results/processed/results.csv` (one row per question, language, model
and method) and saves every raw model reply in `results/raw_outputs/`.

TIP: while testing, set `sample_size` to a small number (like 10) in
`config/config.yaml` so runs are fast and cheap. Raise it to ~500 for the real run.

**Step 3 — analyse the results:**
Open `notebooks/analysis.ipynb` and run the cells, or adapt the functions in
`src/metrics.py`. This produces the metrics table and the reliability diagrams.

## Choosing what to run

Edit `config/config.yaml` to change the languages, the sample size, the list of
models and which confidence methods to use. You should not need to edit the
scripts to change these.

## The two confidence methods

- **verbalized**: the model is asked to state a confidence number. Works on every
  model.
- **logprob**: we read the model's own probability for the answer it chose. Only
  works on models that expose token probabilities (in this config, OpenAI and the
  open-weight models; Gemini and Claude do not).

## Reproducibility

The random seed in `config/config.yaml` fixes which questions are sampled, so the
study reproduces exactly. Raw model replies are saved so you can re-parse them
without paying for the API calls again.

## Data and credit

Belebele: Bandarkar et al., *The Belebele Benchmark: a Parallel Reading
Comprehension Dataset in 122 Language Variants* (2024). Loaded from the
`facebook/belebele` dataset on Hugging Face.

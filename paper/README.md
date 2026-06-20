# Research Protocol

## Research Question

Are Large Language Models (LLMs) less calibrated in low-resource languages such as Uzbek compared to higher-resource languages such as English and Russian?

## Dataset

**Belebele**

### Languages

* English (`eng_Latn`)
* Russian (`rus_Cyrl`)
* Uzbek (`uzn_Latn`)

### Dataset Size

* 900 questions per language
* 500 questions sampled for evaluation

### Sampling Strategy

A fixed benchmark of 500 questions was randomly sampled from the aligned multilingual dataset using:

* Random seed: 42

## Models

* GPT-4o
* Gemini 2.5 Flash
* Qwen 2.5 7B Instruct
* Aya Expanse 8B

## Confidence Method

Verbalized confidence (0–100)

Example:

```text
ANSWER: B
CONFIDENCE: 87
```

## Evaluation Metrics

* Accuracy
* Expected Calibration Error (ECE)
* Brier Score
* Reliability Diagrams

---

# Dataset Validation

## Validation Results

### Language Availability

Successfully loaded and validated:

* English (`eng_Latn`)
* Russian (`rus_Cyrl`)
* Uzbek (`uzn_Latn`)

### Dataset Structure

Each question contains:

* Passage (`flores_passage`)
* Question
* Four answer choices
* Correct answer label

### Questions per Language

| Language | Questions |
| -------- | --------- |
| English  | 900       |
| Russian  | 900       |
| Uzbek    | 900       |

## Alignment Investigation

Several alignment strategies were investigated.

### Rejected Approaches

#### `question_number`

Not globally unique.

#### `link`

Not globally unique.

A single article may contain multiple questions.

### Final Alignment Key

Questions are uniquely identified using:

```text
(link, question_number)
```

This composite key successfully aligned all three languages.

### Alignment Outcome

* 900 aligned multilingual questions
* 488 shared source articles
* Complete alignment across English, Russian, and Uzbek

---

# Milestone 1 Completed

✅ Loaded Belebele dataset

✅ Explored dataset structure

✅ Validated language availability

✅ Investigated multilingual alignment

✅ Identified composite key (`link`, `question_number`)

✅ Built multilingual aligned dataset

✅ Generated fixed benchmark of 500 questions

✅ Prepared dataset for model evaluation

---

# Next Milestone

## Model Evaluation Pipeline

Upcoming tasks:

* Prompt design
* Response parsing
* Model integration
* Confidence extraction
* Calibration metric implementation
* Reliability diagram generation
* Cross-language evaluation

# Research Protocol

## Research Question

Does LLM accuracy and calibration degrade in lower-resource languages, and
does that degradation generalize across language families, or is it specific
to one? A study limited to a single low-resource language cannot distinguish
a general resource-scarcity effect from a quirk of that particular language.
This study tests two independent low-resource/high-resource pairs from
unrelated families: Uzbek/Kazakh (Turkic) and Russian/Macedonian (Slavic),
with English as a shared high-resource reference point.

## Dataset

**Belebele**

### Languages

* English (`eng_Latn`) — high-resource reference
* Russian (`rus_Cyrl`) — high-resource Slavic
* Macedonian (`mkd_Cyrl`) — low-resource Slavic
* Uzbek (`uzn_Latn`) — low-resource Turkic
* Kazakh (`kaz_Cyrl`) — low-resource Turkic

### Dataset Size

* 900 questions aligned across English, Russian, and Uzbek
* 500 questions sampled for evaluation (seed 42)
* Kazakh and Macedonian aligned onto the same 500 questions via the
  `(link, question_number)` composite key, so all five languages are
  evaluated on identical underlying content

### Sampling Strategy

A fixed benchmark of 500 questions was sampled using random seed 42. Kazakh
and Macedonian content was merged onto this same fixed sample via
`src/add_language.py`, which aligns by question identity rather than
resampling, guaranteeing all five languages share the same 500 questions.

## Models

* GPT-4o
* Gemini 2.5 Flash
* Claude Sonnet

## Confidence Method

Verbalized confidence (0-100).

Example:

```text
ANSWER: B
CONFIDENCE: 87
```

A logprob-based confidence method was scoped but not implemented, since it
would require model-specific extraction logic not uniformly available across
the three evaluated providers. Noted as a limitation and future-work
direction.

## Evaluation Metrics

* Accuracy, with 95% bootstrap confidence intervals
* Expected Calibration Error (ECE), with 95% bootstrap confidence intervals
* Brier Score
* Reliability Diagrams
* Paired bootstrap significance tests (1,000 resamples) between every pair of
  languages, within each model, for both accuracy and ECE
* Benjamini-Hochberg FDR correction across all pairwise tests, reported both
  within each metric family and across the full set of tests as a robustness
  check

## Alignment Procedure

Questions are uniquely identified using the composite key:

```text
(link, question_number)
```

`link` alone is not unique (a single article contains multiple questions);
`question_number` alone is not globally unique either. The composite key
successfully aligns all five languages onto one fixed 500-question sample:
488 shared source articles, complete alignment, no missing matches.

## Dataset Structure

Each question contains:

* Passage (`flores_passage`)
* Question
* Four answer choices
* Correct answer label

## Key Results

Accuracy differences across languages remain largely significant after FDR
correction, supporting a structure where English significantly outperforms
all other languages, and Kazakh is significantly worse than Russian and
Macedonian, across all three models. Calibration (ECE) differences are far
less robust: no pairwise ECE comparison survives FDR correction within the
ECE test family, indicating that observed calibration gradients in the raw
data are not statistically distinguishable from noise at this sample size.
This dissociation between accuracy degradation (robust, generalizes across
both language families) and calibration degradation (not statistically
established) is a central finding of the study.

---

# Limitations

* Verbalized confidence only; no logprob-based confidence comparison
* Confidence distributions are near-degenerate (mass concentrated near
  90-100% regardless of correctness), which limits ECE's sensitivity
* Three closed frontier models only; no open-weight model comparison
* Single prompt template, temperature 0, no cross-prompt robustness check
* Five languages across two families; generalization to other language
  families untested
* No pairwise ECE difference survives multiple-comparisons correction; the
  calibration-degradation hypothesis is not conclusively established at this
  sample size

# Next Steps (beyond current scope)

* Logprob-based confidence extraction for at least one provider, as a
  comparison point against verbalized confidence
* Additional low-resource languages or open-weight models, if pursued as
  follow-up work beyond this submission
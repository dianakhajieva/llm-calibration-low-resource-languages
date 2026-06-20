from datasets import load_dataset
import pandas as pd


def load_language(config_name):
    ds = load_dataset("facebook/belebele", config_name)

    rows = []

    for row in ds["test"]:
        rows.append(
            {
                "key": f"{row['link']}__{row['question_number']}",
                "link": row["link"],
                "question_number": row["question_number"],
                "question": row["question"],
                "passage": row["flores_passage"],
                "a1": row["mc_answer1"],
                "a2": row["mc_answer2"],
                "a3": row["mc_answer3"],
                "a4": row["mc_answer4"],
                "correct": row["correct_answer_num"],
            }
        )

    return pd.DataFrame(rows)


eng = load_language("eng_Latn")
rus = load_language("rus_Cyrl")
uzb = load_language("uzn_Latn")

eng = eng.add_prefix("en_")
rus = rus.add_prefix("ru_")
uzb = uzb.add_prefix("uz_")

aligned = (
    eng.merge(
        rus,
        left_on="en_key",
        right_on="ru_key",
        how="inner"
    )
    .merge(
        uzb,
        left_on="en_key",
        right_on="uz_key",
        how="inner"
    )
)

sample = aligned.sample(
    n=500,
    random_state=42
)

sample.to_csv(
    "data/sample/sample_500.csv",
    index=False
)

print("English:", len(eng))
print("Russian:", len(rus))
print("Uzbek:", len(uzb))
print("Aligned rows:", len(aligned))
print("Saved:", len(sample))


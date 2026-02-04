import json
from pathlib import Path

results_dir = Path("results")

symptom_labels = {
    "Sadness": "Sadness",
    "Self-dislike": "Self-Dislike",
    "Loss_of_Pleasure": "Loss of Pleasure",
    "Sense_of_failure": "Past Failure",
    "Pessimism": "Pessimism",
    "Irritability": "Irritability",
    "Feelings_of_worthlessness": "Worthlessness",
    "Guilty_feelings": "Guilty Feelings",
    "Tiredness_or_fatigue": "Tiredness or Fatigue",
    "Crying": "Crying",
    "Social_withdrawal": "Loss of Interest",
    "Suicidal_ideas": "Suicidal Thoughts",
    "Sense_of_punishment": "Punishment Feelings",
    "Indecision": "Indecisiveness",
    "Concentration_difficulty": "Concentration Difficulty",
    "Loss_of_energy": "Loss of Energy",
    "Agitation": "Agitation",
    "Changes_in_appetite": "Changes in Appetite",
    "Change_of_sleep": "Changes in Sleeping Pattern",
    "Self-incrimination": "Self-Criticalness",
    "Loss_of_interest_in_sex": "Loss of Interest in Sex",
}

model_id = "google_gemma-3-4b-it"
methods = ["zs", "icl", "sft", "si"]

data = {}
for method in methods:
    if method == "sft":
        pattern = f"checkpoints_{model_id}_sft_test_sft_eval.json"
    elif method == "icl":
        pattern = f"{model_id}_test_*shot_icl_eval.json"
    else:
        pattern = f"{model_id}_test_{method}_eval.json"

    files = list(results_dir.glob(pattern))
    if files:
        with open(files[0]) as f:
            result = json.load(f)
            data[method] = result["multilabel"]["per_symptom"]

print("\\begin{table}")
print("    \\centering")
print("    \\caption{Per-symptom F1-score on the BDI-Sen test set across inference strategies for the \\textsc{Gemma 3 4B} model. Best results for each symptom are highlighted in \\textbf{bold}, and second-best results are \\underline{underlined}.}")
print("    \\label{tab:bdisen_per_symptom_f1}")
print("    \\begin{tabular}{lcccc}")
print("        \\toprule")
print("        \\textbf{Symptom} & \\textbf{ZS} & \\textbf{ICL} & \\textbf{FT} & \\textbf{SI} \\\\")
print("        \\midrule")

for symptom_key, symptom_label in symptom_labels.items():
    metric_values = []
    for method in methods:
        if method in data and symptom_key in data[method]:
            metric_values.append(data[method][symptom_key]["f1"])
        else:
            metric_values.append(None)

    valid_values = [v for v in metric_values if v is not None]
    if len(valid_values) >= 2:
        sorted_vals = sorted(valid_values, reverse=True)
        best = sorted_vals[0]
        second = sorted_vals[1]
    elif len(valid_values) == 1:
        best = valid_values[0]
        second = None
    else:
        best = second = None

    row_values = []
    for method in methods:
        if method in data and symptom_key in data[method]:
            val = data[method][symptom_key]["f1"]
            formatted = f"{val:.3f}"

            if val == best:
                formatted = f"\\textbf{{{formatted}}}"
            elif val == second:
                formatted = f"\\underline{{{formatted}}}"

            row_values.append(formatted)
        else:
            row_values.append("")

    print(f"        {symptom_label:30s} & {' & '.join(row_values)} \\\\")

print("        \\midrule")

avg_values = []
for method in methods:
    if method in data:
        f1_scores = [data[method][symptom_key]["f1"] for symptom_key in symptom_labels.keys() if symptom_key in data[method]]
        if f1_scores:
            avg_f1 = sum(f1_scores) / len(f1_scores)
            avg_values.append(avg_f1)
        else:
            avg_values.append(None)
    else:
        avg_values.append(None)

valid_avg_values = [v for v in avg_values if v is not None]
if len(valid_avg_values) >= 2:
    sorted_vals = sorted(valid_avg_values, reverse=True)
    best = sorted_vals[0]
    second = sorted_vals[1]
elif len(valid_avg_values) == 1:
    best = valid_avg_values[0]
    second = None
else:
    best = second = None

formatted_avg_values = []
for val in avg_values:
    if val is not None:
        formatted = f"{val:.3f}"
        if val == best:
            formatted = f"\\textbf{{{formatted}}}"
        elif val == second:
            formatted = f"\\underline{{{formatted}}}"
        formatted_avg_values.append(formatted)
    else:
        formatted_avg_values.append("--")

print(f"        \\textit{{Raw average}} & {' & '.join(formatted_avg_values)} \\\\")

print("        \\bottomrule")
print("    \\end{tabular}")
print("\\end{table}")

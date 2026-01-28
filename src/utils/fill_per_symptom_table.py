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

model_id = "google_gemma-3-4b"
methods = ["zs", "icl", "sft", "rcl"]

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

print("\\begin{table*}")
print("    \\centering")
print("    \\caption{Per-symptom classification performance on the BDI-Sen test set across inference strategies for the \\textsc{Gemma 3 4B} model. Best results for each symptom and metric are highlighted in \\textbf{bold}, and second-best results are \\underline{underlined}.}")
print("    \\label{tab:bdisen_per_symptom}")
print("    \\begin{tabular}{lcccc:cccc:cccc}")
print("        \\toprule")
print("        \\multirow{2}{*}{\\textbf{Symptom}}")
print("        & \\multicolumn{4}{c}{\\textbf{Precision (P)}}")
print("        & \\multicolumn{4}{c}{\\textbf{Recall (R)}}")
print("        & \\multicolumn{4}{c}{\\textbf{F1-score (F1)}} \\\\")
print("        \\cmidrule(lr){2-5}")
print("        \\cmidrule(lr){6-9}")
print("        \\cmidrule(lr){10-13}")
print("        ")
print("        & \\textbf{ZS} & \\textbf{ICL} & \\textbf{FT} & \\textbf{SI}")
print("        & \\textbf{ZS} & \\textbf{ICL} & \\textbf{FT} & \\textbf{SI}")
print("        & \\textbf{ZS} & \\textbf{ICL} & \\textbf{FT} & \\textbf{SI} \\\\")
print("        \\midrule")
print("        ")

for symptom_key, symptom_label in symptom_labels.items():
    row_values = []

    for metric in ["precision", "recall", "f1"]:
        metric_values = []
        for method in methods:
            if method in data and symptom_key in data[method]:
                metric_values.append(data[method][symptom_key][metric])
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

        for method in methods:
            if method in data and symptom_key in data[method]:
                val = data[method][symptom_key][metric]
                formatted = f"{val:.3f}"

                if val == best:
                    formatted = f"\\textbf{{{formatted}}}"
                elif val == second:
                    formatted = f"\\underline{{{formatted}}}"

                row_values.append(formatted)
            else:
                row_values.append("")

    print(f"        {symptom_label:30s} & {' & '.join(row_values)} \\\\")

print("        ")
print("        \\bottomrule")
print("    \\end{tabular}")
print("\\end{table*}")

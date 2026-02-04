import json
from pathlib import Path

results_dir = Path("results/cross_domain")

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

disease_labels = {
    "depression": "Depression",
    "bipolar_disorder": "Bipolar Disorder",
    "eating_disorder": "Eating Disorder",
}

model_id = "google_gemma-3-4b-it"
methods = ["zs", "icl", "sft", "si"]

data = {}
for method in methods:
    if method == "sft":
        pattern = f"checkpoints_{model_id}_sft_test_sft_psysym_eval.json"
    elif method == "icl":
        pattern = f"{model_id}_test_*shot_icl_psysym_eval.json"
    else:
        pattern = f"{model_id}_test_{method}_psysym_eval.json"

    files = list(results_dir.glob(pattern))
    if files:
        with open(files[0]) as f:
            result = json.load(f)
            data[method] = result.get("per_disease", {})

disease_symptom_map = {}
for disease_key in disease_labels.keys():
    disease_symptom_map[disease_key] = set()
    for symptom_key in symptom_labels.keys():
        for method in methods:
            if (method in data and
                disease_key in data[method] and
                symptom_key in data[method][disease_key].get("per_symptom", {})):
                symptom_data = data[method][disease_key]["per_symptom"][symptom_key]
                if symptom_data.get("tp", 0) + symptom_data.get("fn", 0) > 0:
                    disease_symptom_map[disease_key].add(symptom_key)
                    break

num_diseases = len(disease_labels)
tabular_spec = "l" + "cccc:" * (num_diseases - 1) + "cccc"

print("\\begin{table*}")
print("    \\centering")
print("    \\caption{Per-symptom F1-score on the PsySym test set across diseases and inference strategies for the \\textsc{Gemma 3 4B} model. Best results for each symptom-disease pair are highlighted in \\textbf{bold}, and second-best are \\underline{underlined}. -- indicates symptom not present in disease.}")
print("    \\label{tab:psysym_per_symptom}")
print(f"    \\begin{{tabular}}{{{tabular_spec}}}")
print("        \\toprule")

header_line1 = "        \\multirow{2}{*}[-2pt]{\\textbf{Symptom}}"
for disease_key, disease_label in disease_labels.items():
    header_line1 += f" & \\multicolumn{{4}}{{c}}{{\\textbf{{{disease_label}}}}}"
header_line1 += " \\\\"
print(header_line1)

cmidrule_line = "        "
for i, disease_key in enumerate(disease_labels.keys(), start=1):
    start_col = 2 + (i - 1) * 4
    end_col = start_col + 3
    cmidrule_line += f"\\cmidrule(lr){{{start_col}-{end_col}}}"
print(cmidrule_line)

method_line = "        "
for _ in disease_labels.keys():
    method_line += " & \\textbf{ZS} & \\textbf{ICL} & \\textbf{SFT} & \\textbf{SI}"
method_line += " \\\\"
print(method_line)
print("        \\midrule")

for symptom_key, symptom_label in symptom_labels.items():
    row_values = []
    for disease_key in disease_labels.keys():
        if symptom_key not in disease_symptom_map[disease_key]:
            row_values.extend(["--"] * 4)
        else:
            metric_values = []
            for method in methods:
                if (method in data and
                    disease_key in data[method] and
                    symptom_key in data[method][disease_key].get("per_symptom", {})):
                    metric_values.append(data[method][disease_key]["per_symptom"][symptom_key]["f1"])
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
                if (method in data and
                    disease_key in data[method] and
                    symptom_key in data[method][disease_key].get("per_symptom", {})):
                    val = data[method][disease_key]["per_symptom"][symptom_key]["f1"]
                    formatted = f"{val:.3f}"

                    if val == best:
                        formatted = f"\\textbf{{{formatted}}}"
                    elif val == second:
                        formatted = f"\\underline{{{formatted}}}"

                    row_values.append(formatted)
                else:
                    row_values.append("--")

    print(f"        {symptom_label:30s} & {' & '.join(row_values)} \\\\")

print("        \\midrule")

avg_row_values = []
for disease_key in disease_labels.keys():
    disease_avg_values = []
    for method in methods:
        if method in data and disease_key in data[method]:
            f1_scores = []
            for symptom_key in symptom_labels.keys():
                if symptom_key in disease_symptom_map[disease_key]:
                    if symptom_key in data[method][disease_key].get("per_symptom", {}):
                        f1_scores.append(data[method][disease_key]["per_symptom"][symptom_key]["f1"])

            if f1_scores:
                avg_f1 = sum(f1_scores) / len(f1_scores)
                disease_avg_values.append(avg_f1)
            else:
                disease_avg_values.append(None)
        else:
            disease_avg_values.append(None)

    valid_values = [v for v in disease_avg_values if v is not None]
    if len(valid_values) >= 2:
        sorted_vals = sorted(valid_values, reverse=True)
        best = sorted_vals[0]
        second = sorted_vals[1]
    elif len(valid_values) == 1:
        best = valid_values[0]
        second = None
    else:
        best = second = None

    for val in disease_avg_values:
        if val is not None:
            formatted = f"{val:.3f}"
            if val == best:
                formatted = f"\\textbf{{{formatted}}}"
            elif val == second:
                formatted = f"\\underline{{{formatted}}}"
            avg_row_values.append(formatted)
        else:
            avg_row_values.append("--")

print(f"        \\textit{{Raw average}} & {' & '.join(avg_row_values)} \\\\")

print("        \\bottomrule")
print("    \\end{tabular}")
print("\\end{table*}")

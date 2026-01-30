import json
from pathlib import Path

results_dir = Path("results")

models = {
    "google_gemma-3-4b-it": "Gemma 3 4B",
    "google_gemma-3-12b-it": "Gemma 3 12B",
    "meta-llama_Llama-3.2-3B-Instruct": "Llama 3.2 3B",
    "meta-llama_Llama-3.1-8B-Instruct": "Llama 3.1 8B",
    "microsoft_Phi-4-mini-instruct": "Phi-4 Mini",
    "microsoft_phi-4": "Phi-4",
    "Qwen_Qwen3-4B-Instruct-2507": "Qwen3 4B",
    "Qwen_Qwen3-14B": "Qwen3 14B",
}

methods = {
    "zs": "ZS",
    "icl": "ICL",
    "sft": "FT",
    "si": "SI",
    "osi": "OSI",
}

data = {}
for model_id in models:
    data[model_id] = {}
    for method in methods:
        if method == "sft":
            pattern = f"checkpoints_{model_id}_sft_test_sft_eval.json"
        elif method == "icl":
            pattern = f"{model_id}_test_*shot_icl_eval.json"
        elif method == "osi":
            pattern = f"*_test_*osi*_eval.json"
        else:
            pattern = f"{model_id}_test_{method}_eval.json"

        files = list(results_dir.glob(pattern))
        if method == "osi" and files:
            if "gemma-3-4b" in model_id.lower():
                files = [f for f in files if "gemma3_4b" in f.name.lower()]
            elif "gemma-3-12b" in model_id.lower():
                files = [f for f in files if "gemma3_12b" in f.name.lower()]
            elif "llama-3.2-3b" in model_id.lower():
                files = [f for f in files if "llama32_3b" in f.name.lower()]
            elif "llama-3.1-8b" in model_id.lower():
                files = [f for f in files if "llama31_8b" in f.name.lower()]
            elif "phi-4-mini" in model_id.lower():
                files = [f for f in files if "phi4_mini" in f.name.lower() or "phi4mini" in f.name.lower()]
            elif "phi-4" in model_id.lower():
                files = [f for f in files if "phi4" in f.name.lower() and "mini" not in f.name.lower()]
            elif "qwen3-4b" in model_id.lower():
                files = [f for f in files if "qwen3_4b" in f.name.lower()]
            elif "qwen3-14b" in model_id.lower():
                files = [f for f in files if "qwen3_14b" in f.name.lower()]
            else:
                files = []
        if files:
            with open(files[0]) as f:
                result = json.load(f)
                weighted = result["multilabel"]["weighted"]
                data[model_id][method] = {
                    "f1": weighted["f1"],
                }

print("\\begin{table}")
print("    \\centering")
print("    \\caption{Weighted F1-score for multilabel classification across different model sizes and inference strategies on the BDI-Sen test set. \\textbf{Bold}/\\underline{underlined} indicate best/second-best per row (strategy comparison). $^\\dagger$/$^\\ddagger$ indicate best/second-best per column (model comparison).}")
print("    \\label{tab:bdi_sen_overall_f1}")
print("    \\begin{tabular}{lccccc}")
print("    \\toprule")
print("    \\textbf{Model} & \\textbf{ZS} & \\textbf{ICL} & \\textbf{FT} & \\textbf{SI} & \\textbf{OSI} \\\\")
print("    \\midrule")

metric = "f1"

col_best = {}
for m in ["zs", "icl", "sft", "si", "osi"]:
    col_values = []
    for model_id in models:
        if m in data[model_id]:
            col_values.append(data[model_id][m][metric])

    if len(col_values) >= 2:
        sorted_col = sorted(col_values, reverse=True)
        col_best[m] = {"best": sorted_col[0], "second": sorted_col[1]}
    else:
        col_best[m] = {"best": None, "second": None}

for model_id, model_name in models.items():
    row_values = []
    for m in ["zs", "icl", "sft", "si", "osi"]:
        if m in data[model_id]:
            row_values.append(data[model_id][m][metric])
        else:
            row_values.append(None)

    valid_row_values = [v for v in row_values if v is not None]
    if len(valid_row_values) >= 2:
        sorted_row = sorted(valid_row_values, reverse=True)
        best_row = sorted_row[0]
        second_row = sorted_row[1]
    elif len(valid_row_values) == 1:
        best_row = valid_row_values[0]
        second_row = None
    else:
        best_row = second_row = None

    formatted_values = []
    for m in ["zs", "icl", "sft", "si", "osi"]:
        if m in data[model_id]:
            val = data[model_id][m][metric]
            formatted = f"{val:.3f}"

            is_best_row = (val == best_row) if best_row is not None else False
            is_second_row = (val == second_row) if second_row is not None else False

            if is_best_row:
                formatted = f"\\textbf{{{formatted}}}"
            elif is_second_row:
                formatted = f"\\underline{{{formatted}}}"

            if val == col_best[m]["best"]:
                formatted = f"{formatted}$^\\dagger$"
            elif val == col_best[m]["second"]:
                formatted = f"{formatted}$^\\ddagger$"

            formatted_values.append(formatted)
        else:
            formatted_values.append("")

    print(f"    \\textsc{{{model_name}}} & {formatted_values[0]} & {formatted_values[1]} & {formatted_values[2]} & {formatted_values[3]} & {formatted_values[4]} \\\\")

print("    \\bottomrule")
print("    \\end{tabular}")
print("\\end{table}")

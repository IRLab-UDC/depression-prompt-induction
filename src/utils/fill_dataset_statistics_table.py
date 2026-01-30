import json
from pathlib import Path

stats_file = Path("results/dataset_stats.json")

with open(stats_file) as f:
    stats = json.load(f)

print("\\begin{table}")
print("    \\centering")
print("    \\caption{BDI-Sen dataset statistics across splits.}")
print("    \\label{tab:dataset_statistics}")
print("    \\begin{tabular}{lrrr}")
print("    \\toprule")
print("    \\textbf{Statistic} & \\textbf{Train} & \\textbf{Val} & \\textbf{Test} \\\\")
print("    \\midrule")

print(f"    Total samples & {stats['train']['n_samples']} & {stats['val']['n_samples']} & {stats['test']['n_samples']} \\\\")
print(f"    \\quad Positive & {stats['train']['positive_samples']} & {stats['val']['positive_samples']} & {stats['test']['positive_samples']} \\\\")
print(f"    \\quad Hard negative & {stats['train']['hard_negative_samples']} & {stats['val']['hard_negative_samples']} & {stats['test']['hard_negative_samples']} \\\\")
print(f"    \\quad Soft negative & {stats['train']['soft_negative_samples']} & {stats['val']['soft_negative_samples']} & {stats['test']['soft_negative_samples']} \\\\")
print("    \\midrule")
print(f"    Total annotations & {stats['train']['positive_annotations'] + stats['train']['hard_negative_annotations']} & {stats['val']['positive_annotations'] + stats['val']['hard_negative_annotations']} & {stats['test']['positive_annotations'] + stats['test']['hard_negative_annotations']} \\\\")
print(f"    \\quad Positive & {stats['train']['positive_annotations']} & {stats['val']['positive_annotations']} & {stats['test']['positive_annotations']} \\\\")
print(f"    \\quad Hard negative & {stats['train']['hard_negative_annotations']} & {stats['val']['hard_negative_annotations']} & {stats['test']['hard_negative_annotations']} \\\\")
print("    \\midrule")

train_counts = list(stats['train']['symptom_counts'].values())
val_counts = list(stats['val']['symptom_counts'].values())
test_counts = list(stats['test']['symptom_counts'].values())

train_min, train_max, train_avg = min(train_counts), max(train_counts), sum(train_counts) / len(train_counts)
val_min, val_max, val_avg = min(val_counts), max(val_counts), sum(val_counts) / len(val_counts)
test_min, test_max, test_avg = min(test_counts), max(test_counts), sum(test_counts) / len(test_counts)

print(f"    Symptom occurrences \\") 
print(f"    \\quad Min & {train_min} & {val_min} & {test_min} \\\\")
print(f"    \\quad Avg & {train_avg:.1f} & {val_avg:.1f} & {test_avg:.1f} \\\\")
print(f"    \\quad Max & {train_max} & {val_max} & {test_max} \\\\")
print("    \\midrule")
print(f"    Pos/Neg ratio & {stats['train']['pos_neg_ratio']:.1f} & {stats['val']['pos_neg_ratio']:.1f} & {stats['test']['pos_neg_ratio']:.1f} \\\\")

print("    \\bottomrule")
print("    \\end{tabular}")
print("\\end{table}")

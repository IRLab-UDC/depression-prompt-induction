import json
from pathlib import Path

stats_file = Path("results/dataset_stats.json")

with open(stats_file) as f:
    all_stats = json.load(f)

target_diseases = ["depression", "bipolar_disorder", "eating_disorder"]

print("\\begin{table*}")
print("    \\centering")
print("    \\caption{Dataset statistics. BDI-Sen focuses on depression, while PsySym includes multiple diseases. Both use the same 21 BDI symptoms. All datasets include positive samples, hard negatives (symptom-bearing but not matching target disease), and soft negatives (control samples).}")
print("    \\label{tab:dataset_statistics}")
print("    \\begin{tabular}{llrrr}")
print("    \\toprule")
print("    \\textbf{Dataset} & \\textbf{Statistic} & \\textbf{Train} & \\textbf{Val} & \\textbf{Test} \\\\")
print("    \\midrule")

for dataset_key, dataset_data in all_stats.items():
    if dataset_key == "bdi_sen_v2":
        stats = dataset_data["overall"]
        dataset_label = "BDI-Sen"

        print(f"    {dataset_label} & Total samples & {stats['train']['n_samples']} & {stats['val']['n_samples']} & {stats['test']['n_samples']} \\\\")
        print(f"    & \\quad Positive & {stats['train']['positive_samples']} & {stats['val']['positive_samples']} & {stats['test']['positive_samples']} \\\\")
        print(f"    & \\quad Hard negative & {stats['train']['hard_negative_samples']} & {stats['val']['hard_negative_samples']} & {stats['test']['hard_negative_samples']} \\\\")
        print(f"    & \\quad Soft negative & {stats['train']['soft_negative_samples']} & {stats['val']['soft_negative_samples']} & {stats['test']['soft_negative_samples']} \\\\")

        train_total_ann = stats['train']['positive_annotations'] + stats['train']['hard_negative_annotations']
        val_total_ann = stats['val']['positive_annotations'] + stats['val']['hard_negative_annotations']
        test_total_ann = stats['test']['positive_annotations'] + stats['test']['hard_negative_annotations']

        print(f"    & Total annotations & {train_total_ann} & {val_total_ann} & {test_total_ann} \\\\")
        print(f"    & \\quad Positive & {stats['train']['positive_annotations']} & {stats['val']['positive_annotations']} & {stats['test']['positive_annotations']} \\\\")
        print(f"    & \\quad Hard negative & {stats['train']['hard_negative_annotations']} & {stats['val']['hard_negative_annotations']} & {stats['test']['hard_negative_annotations']} \\\\")

        train_counts = list(stats['train']['symptom_counts'].values())
        val_counts = list(stats['val']['symptom_counts'].values())
        test_counts = list(stats['test']['symptom_counts'].values())

        train_avg = sum(train_counts) / len(train_counts)
        val_avg = sum(val_counts) / len(val_counts)
        test_avg = sum(test_counts) / len(test_counts)

        print(f"    & Symptom occurrences \\\\")
        print(f"    & \\quad Min & {min(train_counts)} & {min(val_counts)} & {min(test_counts)} \\\\")
        print(f"    & \\quad Avg & {train_avg:.1f} & {val_avg:.1f} & {test_avg:.1f} \\\\")
        print(f"    & \\quad Max & {max(train_counts)} & {max(val_counts)} & {max(test_counts)} \\\\")

        print(f"    & Pos/Neg ratio & {stats['train']['pos_neg_ratio']:.1f} & {stats['val']['pos_neg_ratio']:.1f} & {stats['test']['pos_neg_ratio']:.1f} \\\\")
        print("    \\midrule")
    else:
        for i, disease_key in enumerate(target_diseases):
            if disease_key in dataset_data:
                stats = dataset_data[disease_key]
                disease_name = disease_key.replace("_", " ").title()
                dataset_label = f"PsySym ({disease_name})"

                print(f"    {dataset_label} & Total samples & {stats['train']['n_samples']} & {stats['val']['n_samples']} & {stats['test']['n_samples']} \\\\")
                print(f"    & \\quad Positive & {stats['train']['positive_samples']} & {stats['val']['positive_samples']} & {stats['test']['positive_samples']} \\\\")
                print(f"    & \\quad Hard negative & {stats['train']['hard_negative_samples']} & {stats['val']['hard_negative_samples']} & {stats['test']['hard_negative_samples']} \\\\")
                print(f"    & \\quad Soft negative & {stats['train']['soft_negative_samples']} & {stats['val']['soft_negative_samples']} & {stats['test']['soft_negative_samples']} \\\\")

                train_total_ann = stats['train']['positive_annotations'] + stats['train']['hard_negative_annotations']
                val_total_ann = stats['val']['positive_annotations'] + stats['val']['hard_negative_annotations']
                test_total_ann = stats['test']['positive_annotations'] + stats['test']['hard_negative_annotations']

                print(f"    & Total annotations & {train_total_ann} & {val_total_ann} & {test_total_ann} \\\\")
                print(f"    & \\quad Positive & {stats['train']['positive_annotations']} & {stats['val']['positive_annotations']} & {stats['test']['positive_annotations']} \\\\")
                print(f"    & \\quad Hard negative & {stats['train']['hard_negative_annotations']} & {stats['val']['hard_negative_annotations']} & {stats['test']['hard_negative_annotations']} \\\\")

                train_counts = list(stats['train']['symptom_counts'].values())
                val_counts = list(stats['val']['symptom_counts'].values())
                test_counts = list(stats['test']['symptom_counts'].values())

                if train_counts and val_counts and test_counts:
                    train_avg = sum(train_counts) / len(train_counts)
                    val_avg = sum(val_counts) / len(val_counts)
                    test_avg = sum(test_counts) / len(test_counts)

                    print(f"    & Symptom occurrences \\\\")
                    print(f"    & \\quad Min & {min(train_counts)} & {min(val_counts)} & {min(test_counts)} \\\\")
                    print(f"    & \\quad Avg & {train_avg:.1f} & {val_avg:.1f} & {test_avg:.1f} \\\\")
                    print(f"    & \\quad Max & {max(train_counts)} & {max(val_counts)} & {max(test_counts)} \\\\")

                print(f"    & Pos/Neg ratio & {stats['train']['pos_neg_ratio']:.1f} & {stats['val']['pos_neg_ratio']:.1f} & {stats['test']['pos_neg_ratio']:.1f} \\\\")

                if i < len(target_diseases) - 1:
                    print("    \\midrule")

print("    \\bottomrule")
print("    \\end{tabular}")
print("\\end{table*}")

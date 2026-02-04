import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

SUBPLOT_WIDTH = 4.7
SUBPLOT_HEIGHT = 11.5
LINE_WIDTH = 2
MARKER_SIZE = 6
FILL_ALPHA = 0.25
LABEL_SIZE = 14
TITLE_SIZE = 20
SUPTITLE_SIZE = 22
LABEL_MAX_LEN = 15
LOG_TICKS = [0, 1, 5, 10, 25, 50, 100]
FP_COLOR = "#3b82f6"
FN_COLOR = "#3b82f6"
TICK_MARKERS = ['Ⅰ', 'Ⅱ', 'Ⅲ', 'Ⅳ', 'Ⅴ', 'Ⅵ', 'Ⅶ', 'Ⅷ', 'Ⅸ', 'Ⅹ',
                'Ⅺ', 'Ⅻ', 'ⅩⅢ', 'ⅩⅣ', 'ⅩⅤ', 'ⅩⅥ', 'ⅩⅦ', 'ⅩⅧ', 'ⅩⅨ', 'ⅩⅩ', 'ⅩⅪ']


def load_symptom_data(path):
    with open(path) as f:
        data = json.load(f)
    return data["per_symptom"]


def get_max_values(data_dict, symptoms):
    max_fp = max_fn = 0
    for strategy_data in data_dict.values():
        for symptom in symptoms:
            max_fp = max(max_fp, strategy_data[symptom]["fp"])
            max_fn = max(max_fn, strategy_data[symptom]["fn"])
    return max_fp, max_fn


def get_log_ticks(max_val):
    ticks = [t for t in LOG_TICKS if t <= max_val] + [max_val]
    return [np.log1p(t) for t in ticks], [str(t) for t in ticks]


def configure_polar_plot(ax, angles, log_values, num_symptoms, max_val, strategy, color, marker, show_title=True):
    ax.plot(angles, log_values, f'{marker}-', linewidth=LINE_WIDTH, color=color, markersize=MARKER_SIZE)
    ax.fill(angles, log_values, alpha=FILL_ALPHA, color=color)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([TICK_MARKERS[i] for i in range(num_symptoms)], size=LABEL_SIZE)
    ax.set_ylim(0, np.log1p(max_val))
    tick_vals, tick_labels = get_log_ticks(max_val)
    ax.set_yticks(tick_vals)
    ax.set_yticklabels(tick_labels, size=LABEL_SIZE)
    if show_title:
        ax.set_title(strategy, fontsize=TITLE_SIZE, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.6)
    for spine in ax.spines.values():
        spine.set_visible(False)


def abbreviate_symptom(symptom):
    abbreviations = {
        "Sadness": "Sad.",
        "Self-dislike": "Self-Dis.",
        "Loss_of_Pleasure": "Loss Pleas.",
        "Sense_of_failure": "Past Fail.",
        "Pessimism": "Pess.",
        "Irritability": "Irrit.",
        "Feelings_of_worthlessness": "Worthless.",
        "Guilty_feelings": "Guilt.",
        "Tiredness_or_fatigue": "Tired.",
        "Crying": "Cry.",
        "Social_withdrawal": "Loss Int.",
        "Suicidal_ideas": "Suicidal.",
        "Sense_of_punishment": "Punish.",
        "Indecision": "Indecis.",
        "Concentration_difficulty": "Concentr.",
        "Loss_of_energy": "Loss Energy.",
        "Agitation": "Agit.",
        "Changes_in_appetite": "Appetite.",
        "Change_of_sleep": "Sleep.",
        "Self-incrimination": "Self-Crit.",
        "Loss_of_interest_in_sex": "Loss Sex.",
    }
    return abbreviations.get(symptom, symptom[:10])


def plot_radial_log(data_dict, output_path):
    strategies = list(data_dict.keys())
    symptoms = list(next(iter(data_dict.values())).keys())
    max_fp, max_fn = get_max_values(data_dict, symptoms)

    fig, axes = plt.subplots(2, len(strategies),
                            figsize=(SUBPLOT_WIDTH * len(strategies), SUBPLOT_HEIGHT),
                            subplot_kw=dict(projection='polar'))

    angles = np.linspace(0, 2 * np.pi, len(symptoms), endpoint=False).tolist() + [0]

    for idx, strategy in enumerate(strategies):
        fp_values = [data_dict[strategy][s]["fp"] for s in symptoms]
        fn_values = [data_dict[strategy][s]["fn"] for s in symptoms]

        fp_log = [np.log1p(v) for v in fp_values] + [np.log1p(fp_values[0])]
        fn_log = [np.log1p(v) for v in fn_values] + [np.log1p(fn_values[0])]

        configure_polar_plot(axes[0, idx], angles, fp_log, len(symptoms), max_fp, strategy, FP_COLOR, 'o', show_title=True)
        configure_polar_plot(axes[1, idx], angles, fn_log, len(symptoms), max_fn, strategy, FN_COLOR, 'o', show_title=False)

    fig.text(-0.015, 0.75, 'FP', fontsize=SUPTITLE_SIZE, fontweight='bold', va='center', rotation=90)
    fig.text(-0.015, 0.34, 'FN', fontsize=SUPTITLE_SIZE, fontweight='bold', va='center', rotation=90)

    plt.tight_layout(rect=[0, 0.12, 1, 1])

    legend_items = [f'{TICK_MARKERS[i]}: {abbreviate_symptom(s)}' for i, s in enumerate(symptoms)]

    from matplotlib.font_manager import FontProperties
    font_prop = FontProperties(family='sans-serif', size=16)

    x_start = 0.07
    col_width = 0.13
    y_positions = [0.10, 0.075, 0.05]

    for row_idx in range(3):
        for col_idx in range(7):
            idx = row_idx * 7 + col_idx
            if idx < len(legend_items):
                x_pos = x_start + col_idx * col_width
                y_pos = y_positions[row_idx]
                fig.text(x_pos, y_pos, legend_items[idx], fontproperties=font_prop, va='center', ha='left')

    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zs", required=True)
    parser.add_argument("--icl", required=True)
    parser.add_argument("--sft", required=True)
    parser.add_argument("--si", required=True)
    parser.add_argument("--output", default="results/plots/radial_log.pdf")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data_dict = {
        "ZS": load_symptom_data(args.zs),
        "ICL": load_symptom_data(args.icl),
        "SFT": load_symptom_data(args.sft),
        "SI": load_symptom_data(args.si),
    }

    plot_radial_log(data_dict, output_path)


if __name__ == "__main__":
    main()

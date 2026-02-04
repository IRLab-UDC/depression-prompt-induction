import json
import csv
from pathlib import Path
from collections import defaultdict

DEPRESSION_SYMPTOMS = {
    'Depressed_Mood', 'pessimism', 'Worthlessness_and_guilty',
    'loss_of_interest_or_motivation', 'Suicidal_ideas', 'Indecisiveness',
    'Decreased_energy_tiredness_fatigue', 'sleep_disturbance',
    'Anger_Irritability', 'Hyperactivity_agitation', 'weight_and_appetite_change',
    'Genitourinary_symptoms', 'Inattention', 'poor_memory'
}

ALL_BDI_SYMPTOMS = [
    'Sadness', 'Pessimism', 'Sense_of_failure', 'Loss_of_Pleasure',
    'Guilty_feelings', 'Sense_of_punishment', 'Self-dislike', 'Self-incrimination',
    'Suicidal_ideas', 'Crying', 'Agitation', 'Social_withdrawal',
    'Indecision', 'Feelings_of_worthlessness', 'Loss_of_energy',
    'Change_of_sleep', 'Irritability', 'Changes_in_appetite',
    'Concentration_difficulty', 'Tiredness_or_fatigue', 'Loss_of_interest_in_sex'
]

def load_kb_to_bdi_mapping():
    mapping_path = Path('data/psysym/kb_to_bdi_mapping.json')
    with open(mapping_path) as f:
        return json.load(f)

def parse_psysym_csv(csv_path, output_path, kb_to_bdi):
    import random
    random.seed(42)

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        sentence_data = defaultdict(lambda: {
            'diseases': [],
            'kb_symptoms': set(),
            'metadata': []
        })
        control_candidates = []

        for row in reader:
            sentence = row['sentence']
            disease = row['disease']

            kb_symptoms = []
            for symptom in reader.fieldnames:
                if symptom in ['subreddit_id', 'post_id', 'sentence_id', 'disease', 'sentence', 'uncertain']:
                    continue

                if symptom not in DEPRESSION_SYMPTOMS:
                    continue

                if row[symptom] == '1':
                    kb_symptoms.append(symptom)

            if kb_symptoms:
                sentence_data[sentence]['diseases'].append(disease)
                sentence_data[sentence]['kb_symptoms'].update(kb_symptoms)
                sentence_data[sentence]['metadata'].append({
                    'subreddit_id': row['subreddit_id'],
                    'post_id': row['post_id'],
                    'sentence_id': row['sentence_id'],
                    'disease': disease
                })
            elif disease == 'control':
                control_sample = {
                    'sentence': sentence,
                    'subreddit_id': row['subreddit_id'],
                    'post_id': row['post_id'],
                    'sentence_id': row['sentence_id']
                }
                control_candidates.append(control_sample)

    symptom_samples = []
    for sentence, data in sentence_data.items():
        labels = {bdi_symptom: 0 for bdi_symptom in ALL_BDI_SYMPTOMS}

        kb_symptoms_list = sorted(data['kb_symptoms'])
        for kb_symptom in kb_symptoms_list:
            for bdi_symptom in kb_to_bdi[kb_symptom]:
                labels[bdi_symptom] = 1

        sample = {
            'sentence': sentence,
            'is_control': False,
            'labels': labels,
            'kb_symptoms': kb_symptoms_list,
            'diseases': sorted(list(set(data['diseases']))),
            'metadata': data['metadata']
        }
        symptom_samples.append(sample)

    seen_control_sentences = set()
    unique_controls = []
    for ctrl in control_candidates:
        if ctrl['sentence'] not in seen_control_sentences:
            labels = {bdi_symptom: 0 for bdi_symptom in ALL_BDI_SYMPTOMS}
            control_sample = {
                'sentence': ctrl['sentence'],
                'is_control': True,
                'labels': labels,
                'kb_symptoms': [],
                'diseases': ['control'],
                'metadata': [{
                    'subreddit_id': ctrl['subreddit_id'],
                    'post_id': ctrl['post_id'],
                    'sentence_id': ctrl['sentence_id'],
                    'disease': 'control'
                }]
            }
            unique_controls.append(control_sample)
            seen_control_sentences.add(ctrl['sentence'])

    n_control_needed = len(symptom_samples) * 5
    selected_controls = random.sample(unique_controls, min(n_control_needed, len(unique_controls)))

    all_samples = symptom_samples + selected_controls
    random.shuffle(all_samples)

    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    multi_disease = sum(1 for s in symptom_samples if len(s['diseases']) > 1)
    print(f"  Symptom samples: {len(symptom_samples)} ({multi_disease} appearing in multiple diseases)")
    print(f"  Control samples: {len(selected_controls)} (from {len(unique_controls)} available)")
    print(f"  Total: {len(all_samples)} samples")
    print(f"  Output: {output_path.name}")

def main():
    data_dir = Path('data/psysym')
    output_dir = Path('data/psysym')

    kb_to_bdi = load_kb_to_bdi_mapping()

    splits = ['train', 'val', 'test']

    for split in splits:
        csv_path = data_dir / f'{split}.csv'
        output_path = output_dir / f'{split}.jsonl'

        if csv_path.exists():
            print(f"Processing {csv_path.name}:")
            parse_psysym_csv(csv_path, output_path, kb_to_bdi)
        else:
            print(f"Warning: {csv_path} not found")

if __name__ == '__main__':
    main()

def classification_metric(example, prediction, trace=None):
    correct = prediction.answer == example.answer
    return float(correct) if trace is None else correct


def weighted_classification_metric(example, prediction, trace=None):
    pred_yes = prediction.answer == "YES"
    gold_yes = example.answer == "YES"

    if trace is None:
        if pred_yes and gold_yes:
            return 1.0
        elif not pred_yes and not gold_yes:
            return 0.9
        elif pred_yes and not gold_yes:
            return 0.3
        else:
            return 0.0
    return pred_yes == gold_yes

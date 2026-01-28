def classification_metric(example, prediction, trace=None):
    """
    Simple binary accuracy metric.
    """
    correct = prediction.answer == example.answer

    if trace is None:
        return float(correct)
    else:
        return correct


def weighted_classification_metric(example, prediction, trace=None):
    """
    Weighted metric that penalizes false negatives more than false positives.
    Useful when missing a symptom (false negative) is worse than over-predicting.
    """
    correct = prediction.answer == example.answer

    if trace is None:
        if correct:
            if example.answer == "YES":
                return 1.0
            else:
                return 0.8
        else:
            if example.answer == "YES":
                return 0.0
            else:
                return 0.2
    else:
        return correct

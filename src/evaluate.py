"""Shared evaluation utilities.

Macro-F1 is the official metric for this project and must be used
consistently across every notebook, not just the final evaluation.
"""

from sklearn.metrics import classification_report, confusion_matrix, f1_score


def macro_f1(y_true, y_pred):
    """Compute macro-averaged F1, the official metric for this project."""
    return f1_score(y_true, y_pred, average="macro")


def evaluate_predictions(y_true, y_pred, labels=("OTHER", "OFFENSE")):
    """Bundle the metrics used to report model performance."""
    return {
        "macro_f1": macro_f1(y_true, y_pred),
        "report": classification_report(y_true, y_pred, labels=labels),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
    }

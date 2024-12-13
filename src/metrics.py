
"""
This file contains functions to calculate various evaluation metrics for the models.
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, auc

def compute_accuracy(y_true, y_pred):
    """Computes the accuracy."""
    return accuracy_score(y_true, y_pred)

def compute_precision(y_true, y_pred, average='weighted'):
    """Computes the precision (weighted average by default)."""
    return precision_score(y_true, y_pred, average=average)

def compute_recall(y_true, y_pred, average='weighted'):
    """Computes the recall (weighted average by default)."""
    return recall_score(y_true, y_pred, average=average)

def compute_f1_score(y_true, y_pred, average='weighted'):
    """Computes the F1-score (weighted average by default)."""
    return f1_score(y_true, y_pred, average=average)

def compute_confusion_matrix(y_true, y_pred):
    """Computes the confusion matrix."""
    return confusion_matrix(y_true, y_pred)

def compute_roc_auc(y_true, y_prob):
    """Computes the area under the ROC curve (AUC)."""
    try:
        return roc_auc_score(y_true, y_prob)
    except ValueError:
        print("Error: ROC AUC score could not be computed. Check your labels and predictions.")
        return None

def compute_roc_curve_data(y_true, y_prob):
    """Computes data for ROC curve plotting."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc


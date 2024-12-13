

"""This file contains functions to generate various plots for visualizing the model's performance.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from sklearn.metrics import roc_curve, auc
import torch
import torch.nn as nn
import torch.optim as optim
import os
from sklearn.metrics import multilabel_confusion_matrix

def plot_single_confusion_matrix(model, test_dataloader, device, model_name, save_dir):
    """Plots a confusion matrix."""

    model.eval()
    all_test_preds = []
    all_test_labels = []
    with torch.no_grad():
        for batch in test_dataloader:
            images = batch['Image'].to(device)
            labels = batch['Label'].to(device)
            preds = torch.sigmoid(model(images)) >= 0.5
            all_test_preds.append(preds.cpu())
            all_test_labels.append(labels.cpu())

    test_preds = torch.cat(all_test_preds).numpy()
    test_labels = torch.cat(all_test_labels).numpy()

    report = classification_report(test_labels, test_preds, target_names=[f"Label {i+1}" for i in range(15)])
    print(report)

    cm = confusion_matrix(test_labels.ravel(), test_preds.ravel())
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.savefig(os.path.join(save_dir, f"single_confusion_matrix_{model_name}.png"))
    plt.show()

def plot_confusion_matrix(y_true: list, y_pred: list, df: pd.DataFrame, model_name, save_dir):
  '''Takes in a list of predicted labels, list of actual labels, and list of different feature names,
  and returns a grid of confusion matrices, one for each label in the dataset.'''
  # Code block created with assistance from Google Gemini
  label_names = list(df.columns[1:16])
  confusion_matrices = multilabel_confusion_matrix(y_true, y_pred)
  fig, axs = plt.subplots(3, 5, figsize=(10, 5))
  for i, cm in enumerate(confusion_matrices):
      row = i // 5
      col = i % 5
      ax = axs[row, col]
      ax.imshow(cm, cmap=plt.cm.Blues)
      ax.set_title(label_names[i])
      for j in range(len(cm)):
          for k in range(len(cm)):
              text = ax.text(k, j, cm[j, k],
                            ha="center", va="center", color="white" if cm[j, k] > cm.max() / 2 else "black")
  fig.suptitle("Feature Confusion Matrices")
  fig.supxlabel("Predicted")
  fig.supylabel("Actual")
  plt.tight_layout()
  plt.savefig(os.path.join(save_dir, f"confusion_matrix_{model_name}.png"))
  plt.show()

def plot_roc_curves(save_dir, model, test_dataloader, device, model_name, title="Receiver Operating Characteristic (ROC) Curves"):
    """Plots ROC curves for multi-label classification."""
    model.eval()
    all_test_preds_probs = []
    all_test_labels = []
    with torch.no_grad():
        for batch in test_dataloader:
            images = batch['Image'].to(device)
            labels = batch['Label'].to(device)
            preds_probs = torch.sigmoid(model(images))
            all_test_preds_probs.append(preds_probs.cpu())
            all_test_labels.append(labels.cpu())

    test_preds_probs = torch.cat(all_test_preds_probs).numpy()
    test_labels = torch.cat(all_test_labels).numpy()

    plt.figure(figsize=(12, 8))
    for i in range(test_labels.shape[1]):
        fpr, tpr, thresholds = roc_curve(test_labels[:, i], test_preds_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Label {i+1} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title(f"{title} - {model_name}", fontsize=18)
    plt.legend(loc="lower right", fontsize=14)
    plt.savefig(os.path.join(save_dir, f"ROC_{model_name}.png"))
    plt.show()

def plot_training_validation_metrics(save_dir, train_results, model_name, title="Training and Validation Metrics"):
    """Plots training and validation metrics including loss and accuracy."""
    train_losses = train_results['train_loss']
    train_ham_accs = train_results['train_ham_acc']
    train_zero_one_accs = train_results['train_zero_one_acc']
    test_losses = train_results['test_loss']
    test_ham_accs = train_results['test_ham_acc']
    test_zero_one_accs = train_results['test_zero_one_acc']
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 10))

    plt.subplot(3, 1, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, test_losses, label='Validation Loss')
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.title(f'Training and Validation Loss - {model_name}', fontsize=18)
    plt.legend(fontsize=12)

    plt.subplot(3, 1, 2)
    plt.plot(epochs, train_ham_accs, label='Training Hamming Accuracy')
    plt.plot(epochs, test_ham_accs, label='Validation Hamming Accuracy')
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.title(f'Training and Validation Hamming Accuracy - {model_name}', fontsize=18)
    plt.legend(fontsize=12)

    plt.subplot(3, 1, 3)
    plt.plot(epochs, train_zero_one_accs, label='Training Zero-One Accuracy')
    plt.plot(epochs, test_zero_one_accs, label='Validation Zero-One Accuracy')
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.title(f'Training and Validation Zero-One Accuracy - {model_name}', fontsize=18)
    plt.legend(fontsize=12)
    plt.savefig(os.path.join(save_dir, f"training_and_Validation_{model_name}.png"))
    plt.tight_layout()
    plt.show()

def at_the_bar(data, title, file, model_name, set_name, save_dir):
  plt.figure(figsize=(14, 7))
  ax = sns.barplot(x=data.index, y=data.values, palette="coolwarm")
  plt.xticks(rotation=45, ha='right')
  plt.title(title, fontsize=16)
  plt.xlabel('', fontsize=12)

  for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 5), textcoords='offset points')

  plt.tight_layout()
  plt.savefig(os.path.join(save_dir, f"feature_prevalence_{model_name}_{set_name}.png"))
  plt.show()
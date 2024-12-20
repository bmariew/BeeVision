
"""helper_functions.ipynb
Credit for code from PyTorch Deep Learning (Train and Test Loops):
Bourke, Daniel. "engine.py." pytorch-deep-learning, 
https://github.com/mrdbourke/pytorch-deep-learning/blob/main/going_modular/going_modular/engine.py. Accessed 03 November 2024.
"""

import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, auc

# Hamming accuracy

def hamming_accuracy(y_true, y_pred, threshold=0.5):
  """Calculates and returns the hamming accuracy for a given set of ground truth labels (y_true)
  and predicted logits (y_pred)."""

  y_pred_probs = torch.sigmoid(y_pred)
  y_pred_binary = (y_pred_probs >= threshold).float()
  correct_preds = (y_pred_binary == y_true).float()
  hamming_accuracy = correct_preds.mean().item()

  return hamming_accuracy


# Zero one accuracy

def zero_one_accuracy(y_true, y_pred, threshold=0.5):
  """Calculates and returns the zero-one accuracy for a given set of ground truth labels (y_true)
  and predicted logits (y_pred)."""

  y_pred_probs = torch.sigmoid(y_pred)
  y_pred_binary = (y_pred_probs >= threshold).float()
  correct_preds = torch.all(y_pred_binary == y_true, dim=1).float()
  zero_one_accuracy = correct_preds.mean()

  return zero_one_accuracy


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device):
    """Trains a PyTorch model for a single epoch."""

    model.train()
    train_loss, train_ham_acc, train_zero_one_acc = 0, 0, 0

    for batch in dataloader:
        X = batch['Image']
        y = batch['Label']

        X, y = X.to(device), y.to(device)

        # Reshape y to match expected input size
        #y = y.squeeze(1)


        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        train_ham_acc += hamming_accuracy(y_true=y,
                                          y_pred=y_pred,
                                          threshold=0.5)
        train_zero_one_acc += zero_one_accuracy(y_true=y,
                                                y_pred=y_pred,
                                                threshold=0.5)


    train_loss = train_loss / len(dataloader)
    train_ham_acc = train_ham_acc / len(dataloader)
    train_zero_one_acc = train_zero_one_acc / len(dataloader)

    return train_loss, train_ham_acc, train_zero_one_acc

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device):
    """Tests a PyTorch model for a single epoch."""

    model.eval()
    test_loss, test_ham_acc, test_zero_one_acc = 0, 0, 0

    with torch.inference_mode():
        for batch in dataloader:
            X = batch['Image']
            y = batch['Label']

            X, y = X.to(device), y.to(device)
            y = y.squeeze(1)
            test_pred_logits = model(X)
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()
            test_ham_acc += hamming_accuracy(y_true=y,
                                              y_pred=test_pred_logits,
                                              threshold=0.5)
            test_zero_one_acc += zero_one_accuracy(y_true=y,
                                                    y_pred=test_pred_logits,
                                                    threshold=0.5)

    test_loss = test_loss / len(dataloader)
    test_ham_acc = test_ham_acc / len(dataloader)
    test_zero_one_acc = test_zero_one_acc / len(dataloader)

    return test_loss, test_ham_acc, test_zero_one_acc



def train(model:torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device,
          writer=None,
          params=None):
    """Trains and tests a PyTorch model."""

    results = {
        "train_loss": torch.tensor([]),  # Initialize as empty tensors
        "train_ham_acc": torch.tensor([]),
        "train_zero_one_acc": torch.tensor([]),
        "test_loss": torch.tensor([]),
        "test_ham_acc": torch.tensor([]),
        "test_zero_one_acc": torch.tensor([])
    }


    model.to(device)

    for epoch in tqdm(range(epochs)):
        train_loss, train_ham_acc, train_zero_one_acc = train_step(model=model,
                                          dataloader=train_dataloader,
                                          loss_fn=loss_fn,
                                          optimizer=optimizer,
                                          device=device)
        test_loss, test_ham_acc, test_zero_one_acc = test_step(model=model,
          dataloader=test_dataloader,
          loss_fn=loss_fn,
          device=device)

        # Convert to tensors if they are not already
        train_loss = torch.tensor([train_loss])
        train_ham_acc = torch.tensor([train_ham_acc])
        train_zero_one_acc = torch.tensor([train_zero_one_acc])
        test_loss = torch.tensor([test_loss])
        test_ham_acc = torch.tensor([test_ham_acc])
        test_zero_one_acc = torch.tensor([test_zero_one_acc])


        if writer:
            writer.add_scalars(main_tag="Loss",
                               tag_scalar_dict={"train_loss": train_loss.item(),
                                                "test_loss": test_loss.item()},
                               global_step=epoch)
            writer.add_scalars(main_tag="Accuracy",
                               tag_scalar_dict={"train_ham_acc": train_ham_acc.item(),
                                                "train_zero_one_acc": train_zero_one_acc.item(),
                                                "test_ham_acc": test_ham_acc.item(),
                                                "test_zero_one_acc": test_zero_one_acc.item()},
                               global_step=epoch)

        print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss.item():.4f} | "
          f"train_ham_acc: {train_ham_acc.item():.4f} | "
          f"train_zero_one_acc: {train_zero_one_acc.item():.4f} | "
          f"test_loss: {test_loss.item():.4f} | "
          f"test_ham_acc: {test_ham_acc.item():.4f} | "
          f"test_zero_one_acc: {test_zero_one_acc.item():.4f}"
        )

        results["train_loss"] = torch.cat((results["train_loss"], train_loss))
        results["train_ham_acc"] = torch.cat((results["train_ham_acc"], train_ham_acc))
        results["train_zero_one_acc"] = torch.cat((results["train_zero_one_acc"], train_zero_one_acc))
        results["test_loss"] = torch.cat((results["test_loss"], test_loss))
        results["test_ham_acc"] = torch.cat((results["test_ham_acc"], test_ham_acc))
        results["test_zero_one_acc"] = torch.cat((results["test_zero_one_acc"], test_zero_one_acc))

    if writer and params:
      metrics = {"Final Test Loss": test_loss.item()} #Convert to scalar before passing to writer
      writer.add_hparams(hparam_dict=params, metric_dict=metrics)
      writer.close()

    return results



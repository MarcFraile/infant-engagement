import torch
from torch import Tensor


def accuracy(predicted: Tensor, labels: Tensor) -> Tensor:
    """
    Calculate the accuracy (fraction of correctly predicted labels) of a tensor.

    * Keeps result as a tensor.
    """
    return (predicted == labels).float().mean()


def confusion(predicted: Tensor, labels: Tensor, num_classes: int = 2) -> Tensor:
    """
    Calculate the confusion matrix.

    * Assumed shape for both `predicted` and `labels`: `(num_samples,)`.
    * Result given as absolute counts.
    * Result shape `(label, predicted)`.
    * Defaults to binary classification (`num_classes = 2`).
    * Both tensors need to be in the same device.
    """
    assert num_classes       >= 2                , f"Expected at least 2 classes. Found num_classes={num_classes}"
    assert len(labels.shape) == 1                , f"Expected numerical encoding (labels.shape=(num_samples,)). Found labels.shape={labels.shape}"
    assert labels.shape      == predicted.shape  , f"Expected labels and predicted to have the same shape. Found labels.shape={labels.shape}, predicted.shape={predicted.shape}"
    assert labels.device     == predicted.device , f"Expected labels and predicted to be in the same device. Found labels.device={labels.device}, predicted.device={predicted.device}"

    confusion = torch.zeros((num_classes, num_classes), device=labels.device)

    for (l, p) in zip(labels, predicted):
        l = l.long()
        p = p.long()
        assert 0 <= l < num_classes
        assert 0 <= p < num_classes
        confusion[l, p] += 1

    return confusion


def acc_from_conf(confusion: Tensor) -> Tensor:
    """
    Calculate the accuracy from the confusion matrix.
    """
    return confusion.diag().sum() / (confusion.sum() + 1e-6)


def f1(confusion: Tensor) -> Tensor:
    """
    Calculate the F1 score for a binary classification problem.

    * Expected input: one sample of the confusion matrix (shape `(2,2)`).
    """
    assert confusion.shape == (2, 2)

    [[tn, fp], [fn, tp]] = confusion
    return tp / (tp + 0.5 * (fp + fn) + 1e-6)


def balanced_accuracy(confusion: Tensor) -> Tensor:
    """
    Calculate the balanced accuracy, given the `confusion` matrix.

    * Expected input: one sample of the confusion matrix (shape `(num_classes, num_classes)`).
    * For each class c, calculates (correctly predicted c) / (ground truth c).
    * Averages over all classes.
    """
    per_class = confusion.diag() / (confusion.sum(dim=1) + 1e-6)
    return per_class.mean()

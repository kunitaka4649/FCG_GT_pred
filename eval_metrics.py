"""eval metric functions are included"""
import numpy as np
from sklearn.metrics import f1_score


def emr(y_true, y_pred):
    """Exact Match Ratio (EMR)
    The Exact Match Ratio evaluation metric extends the concept the accuracy from the single-label classification problem to a multi-label classification problem.
    One of the drawbacks of using EMR is that is does not account for partially correct labels."""
    n = len(y_true)
    row_indicators = np.all(
        y_true == y_pred, axis=1
    )  # axis = 1 will check for equality along rows.
    for idx, ok in enumerate(row_indicators):
        if ok:
            print(idx + 2)
            print(y_true[idx], y_pred[idx])
    exact_match_count = np.sum(row_indicators)
    return exact_match_count / n



def emr_each_label(y_true, y_pred):
    """Exact Match Ratio (EMR)
    The Exact Match Ratio evaluation metric extends the concept the accuracy from the single-label classification problem to a multi-label classification problem.
    One of the drawbacks of using EMR is that is does not account for partially correct labels."""
    sums = np.zeros(10, dtype=float)
    for idx, labels in enumerate(y_true):
        for j, label in enumerate(labels):
            if y_true[idx][j] == y_pred[idx][j]:
                sums[j] += 1
    sums /= len(y_true)
    return sums


def hamming_loss(y_true, y_pred):
    """
    XOR TT for reference -

    A  B   Output

    0  0    0
    0  1    1
    1  0    1
    1  1    0
    """
    hl_num = np.sum(np.logical_xor(y_true, y_pred))
    hl_den = np.prod(y_true.shape)

    return hl_num / hl_den


def recall_of_each_class(y_true, y_pred):
    """
    Compute recall for each class.
    """
    recall = np.zeros(y_true.shape[1])
    for i in range(y_true.shape[1]):
        recall[i] += np.sum(np.logical_and(y_true[:, i], y_pred[:, i])) / np.sum(
            y_true[:, i]
        )
    return recall


def precision_of_each_class(y_true, y_pred):
    """
    Compute precision for each class.
    """
    precision = np.zeros(y_true.shape[1])
    for i in range(y_true.shape[1]):
        precision[i] += np.sum(np.logical_and(y_true[:, i], y_pred[:, i])) / np.sum(
            y_pred[:, i]
        )
    return precision


def F1Measure(y_true, y_pred):
    return f1_score(y_true=y_true, y_pred=y_pred, average='weighted')


def label_based_micro_precision(y_true, y_pred):

    # compute sum of true positives (tp) across training examples
    # and labels.

    l_prec_num = np.sum(np.logical_and(y_true, y_pred))

    output = ''
    for line in np.logical_and(y_true, y_pred):
        text = ','.join(['1' if ele == True else '0' for ele in line])
        output += text + '\n'
    with open('tmp_mc.csv', 'w') as f:
        f.write(output)

    # compute the sum of tp + fp across training examples and labels
    l_prec_den = np.sum(y_pred)

    # compute micro-averaged precision
    return l_prec_num / l_prec_den


def label_based_micro_recall(y_true, y_pred):

    # compute sum of true positives across training examples and labels.
    l_recall_num = np.sum(np.logical_and(y_true, y_pred))
    # compute sum of tp + fn across training examples and labels
    l_recall_den = np.sum(y_true)

    # compute mirco-average recall
    return l_recall_num / l_recall_den

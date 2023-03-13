import numpy as np
def evaluate_acc(label, pred):
    label = np.array(label).astype(bool)
    pred = np.array(pred)

    threshold = 127.5
    pred = (pred > threshold).astype(bool)

    true_positive = np.sum(np.logical_and(label, pred))
    true_negative = np.sum(np.logical_and(~label, ~pred))
    false_positive = np.sum(np.logical_and(~label, pred))
    false_negative = np.sum(np.logical_and(label, ~pred))

    acc = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)

    return acc
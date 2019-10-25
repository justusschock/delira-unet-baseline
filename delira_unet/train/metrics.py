import numpy as np
import logging

logger = logging.getLogger(__file__)


def compute_stat_score(pred, label, n):
    """Compute metric statistics (true positive, false positive, true negative,
    false negative)
    Parameters
    ----------
    pred : np.ndarray
        network predictions with probability for each class
    label : np.ndarray
        ground truth annotation. Classes are given by numeric value
    n : int
        compute scores for class n
    Returns
    -------
    int
        number of true positives
    int
        number of false positives
    int
        number of true negatives
    int
        number of false negatives
    """
    cls_pred = np.argmax(pred, axis=1)

    tp = ((cls_pred == n) * (label == n)).sum()
    fp = ((cls_pred == n) * (label != n)).sum()
    tn = ((cls_pred != n) * (label != n)).sum()
    fn = ((cls_pred != n) * (label == n)).sum()

    return tp, fp, tn, fn


def dice_score_including_background(pred, label, bg=False, cls_logging=False,
                                    nan_score=0.0, no_fg_score=0.0):
    """
    Compute dice score 1/n_classes * (2*tp)/(2*tp + fp + fn)
    
    Parameters
    ----------
    pred : np.ndarray
        probability for each class
    label : np.ndarray
        ground truth annotation. Classes are given by numeric value
    bg : bool, optional
        compute dice for background class, by default False
    cls_logging : int, optional
        logging for individual class results
    nan_score: float, optional
        if denominator is zero `nan_score`is used instead.
    no_fg_score: float, optional
        if foreground class is not present, `np_fg_score` is sued instead.
    Returns
    -------
    float
        dice score
    """
    if not np.count_nonzero(pred) > 0:
        logger.warning(
            "Prediction only contains zeros. Dice score might be ambigious.")

    # invert background value
    bg = (1 - int(bool(bg)))

    n_classes = pred.shape[1]
    score = 0
    for i in range(bg, n_classes):
        tp, fp, tn, fn = compute_stat_score(pred, label, i)

        if not np.any(label == i):
            # no foreground class
            score_cls = no_fg_score
        elif np.isclose((2 * tp + fp + fn), 0):
            # nan result
            score_cls = nan_score
        else:
            score_cls = (2 * tp) / (2 * tp + fp + fn)

        if cls_logging:
            logger.info({'value': {'value': score_cls,
                                   'name': 'dice_cls_' + str(i)}})
            pass

        score += score_cls
    return score / (n_classes - bg)
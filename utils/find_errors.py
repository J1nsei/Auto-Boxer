from typing import Dict, Set, Tuple
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torchvision

TARGETS_DF_COLUMNS = [
    "target_id",
    "image_id",
    "label_id",
    "xmin",
    "ymin",
    "xmax",
    "ymax",
]
PREDS_DF_COLUMNS = [
    "pred_id",
    "image_id",
    "label_id",
    "xmin",
    "ymin",
    "xmax",
    "ymax",
    "score",
]
ERRORS_DF_COLUMNS = ["pred_id", "target_id", "error_type"]

BACKGROUND_IOU_THRESHOLD = 0.1
FOREGROUND_IOU_THRESHOLD = 0.5


class ErrorType:
    OK = "correct"  # pred -> IoU > foreground; target_label == pred_label; highest score
    CLS = "classification"  # pred -> IoU > foreground; target_label != pred_label
    LOC = "localization"  # pred -> background < IoU < foreground; target_label == pred_label
    CLS_LOC = "cls & loc"  # pred -> background < IoU < foreground; target_label != pred_label
    DUP = "duplicate"  # pred -> background < IoU < foreground; target_label != pred_label
    BKG = "background"  # pred -> IoU > foreground; target_label == pred_label; no highest score
    MISS = "missed"  # target -> No pred with Iou > background


def classify_predictions_errors(
        targets_df: pd.DataFrame,
        preds_df: pd.DataFrame,
        iou_background: float = BACKGROUND_IOU_THRESHOLD,
        iou_foreground: float = FOREGROUND_IOU_THRESHOLD,
) -> pd.DataFrame:
    """Classify predictions

    We assume model is right as much as possible. Thus, in case of doubt
    (i.e matching two targets), a prediction will be first considered
    ErrorType.LOC before ErrorType.CLS.

    :param targets_df: DataFrame with all targets for all images with TARGETS_DF_COLUMNS.
    :param preds_df: DataFrame with all predictions for all images with PREDS_DF_COLUMNS.
    :param iou_background: Minimum IoU for a prediction not to be considered background.
    :param iou_foreground: Minimum IoU for a prediction to be considered foreground.
    :return errors_df: DataFrame with all error information with ERRORS_DF_COLUMNS
    """

    # Provide clarity on expectations and avoid confusing errors down the line
    assert (set(TARGETS_DF_COLUMNS) - set(targets_df.columns)) == set()
    assert (set(PREDS_DF_COLUMNS) - set(preds_df.columns)) == set()

    pred2error = dict()  # {pred_id: ErrorType}
    target2pred = (
        dict()
    )  # {target_id: pred_id}, require iou > iou_foreground & max score
    pred2target = dict()  # {pred_id: target_id}, require iou >= iou_background
    missed_targets = set()  # {target_id}

    # Higher scoring preds take precedence when multiple fulfill criteria
    preds_df = preds_df.sort_values(by="score", ascending=False)

    for image_id, im_preds_df in tqdm(preds_df.groupby("image_id"), desc='Classify errors'):
        # Need to reset index to access dfs with same idx we access
        #   IoU matrix down the line
        im_targets_df = targets_df.query("image_id == @image_id").reset_index(
            drop=True
        )
        im_preds_df = im_preds_df.reset_index(drop=True)

        if im_targets_df.empty:
            pred2error = {**pred2error, **_process_empty_image(im_preds_df)}
        else:
            iou_matrix, iou_label_match_matrix = _compute_iou_matrices(
                im_targets_df, im_preds_df
            )

            # Iterate over all predictions. Higher scores first
            for pred_idx in range(len(im_preds_df)):
                match_found = _match_pred_to_target_with_same_label(
                    pred_idx,
                    pred2error,
                    pred2target,
                    target2pred,
                    iou_label_match_matrix,
                    im_targets_df,
                    im_preds_df,
                    iou_background,
                    iou_foreground,
                )
                if match_found:
                    continue

                _match_pred_wrong_label_or_background(
                    pred_idx,
                    pred2error,
                    pred2target,
                    iou_matrix,
                    im_targets_df,
                    im_preds_df,
                    iou_background,
                    iou_foreground,
                )

    missed_targets = _find_missed_targets(targets_df, pred2target)
    errors_df = _format_errors_as_dataframe(
        pred2error, pred2target, missed_targets
    )
    return errors_df[list(ERRORS_DF_COLUMNS)]


def _process_empty_image(im_preds_df: pd.DataFrame) -> Dict[int, str]:
    """In an image without targets, all predictions represent a background error"""
    return {
        pred_id: ErrorType.BKG for pred_id in im_preds_df["pred_id"].unique()
    }


def _compute_iou_matrices(
        im_targets_df: pd.DataFrame, im_preds_df: pd.DataFrame
) -> Tuple[np.array, np.array]:
    """Compute IoU matrix between all targets and preds in the image

    :param im_targets_df: DataFrame with targets for the image being processed.
    :param im_preds_df: DataFrame with preds for the image being processed.
    :return:
        iou_matrix: Matrix of size (n_targets, n_preds) with IoU between all
            targets & preds
        iou_label_match_matrix: Same as `iou_matrix` but 0 for all target-pred
            pair with different labels (i.e. IoU kept only if labels match).
    """
    # row indexes point to targets, column indexes to predictions
    iou_matrix = iou_matrix = torchvision.ops.box_iou(
        torch.from_numpy(
            im_targets_df[["xmin", "ymin", "xmax", "ymax"]].values
        ),
        torch.from_numpy(im_preds_df[["xmin", "ymin", "xmax", "ymax"]].values),
    ).numpy()

    # boolean matrix with True iff target and pred have the same label
    label_match_matrix = (
            im_targets_df["label_id"].values[:, None]
            == im_preds_df["label_id"].values[None, :]
    )
    # IoU matrix with 0 in all target-pred pairs that have different label
    iou_label_match_matrix = iou_matrix * label_match_matrix
    return iou_matrix, iou_label_match_matrix


def _match_pred_to_target_with_same_label(
        pred_idx: int,
        pred2error: Dict[int, str],
        pred2target: Dict[int, int],
        target2pred: Dict[int, int],
        iou_label_match_matrix: np.array,
        im_targets_df: pd.DataFrame,
        im_preds_df: pd.DataFrame,
        iou_background: float,
        iou_foreground: float,
) -> bool:
    """Try to match `pred_idx` to a target with the same label and identify error (if any)

    If there is a match `pred2error`, `pred2target` and (maybe) `target2pred`
    are modified in place.
    Possible error types found in this function:
        ErrorType.OK, ErrorType.DUP, ErrorType.LOC

    :param pred_idx: Index of prediction based on score (index 0 is maximum score for image).
    :param pred2error: Dict mapping pred_id to error type.
    :param pred2target: Dict mapping pred_id to target_id (if match found with iou above background)
    :param target2pred: Dict mapping target_id to pred_id to pred considered correct (if any).
    :param iou_label_match_matrix: Matrix with size [n_targets, n_preds] with IoU between all preds
        and targets that share label (i.e. IoU = 0 if there is a label missmatch).
    :param im_targets_df: DataFrame with targets for the image being processed.
    :param im_preds_df: DataFrame with preds for the image being processed.
    :param iou_background: Minimum IoU to consider a pred not background for target.
    :param iou_foreground: Minimum IoU to consider a pred foreground for a target.
    :return matched: Whether or not there was a match and we could identify the pred error.
    """
    # Find highest overlapping target for pred processed
    target_idx = np.argmax(iou_label_match_matrix[:, pred_idx])
    iou = np.max(iou_label_match_matrix[:, pred_idx])
    target_id = im_targets_df.at[target_idx, "target_id"]
    pred_id = im_preds_df.at[pred_idx, "pred_id"]

    matched = False
    if iou >= iou_foreground:
        pred2target[pred_id] = target_id
        # Check if another prediction is already the match for target to
        #   identify duplicates
        if target2pred.get(target_id) is None:
            target2pred[target_id] = pred_id
            pred2error[pred_id] = ErrorType.OK
        else:
            pred2error[pred_id] = ErrorType.DUP
        matched = True

    elif iou_background <= iou < iou_foreground:
        pred2target[pred_id] = target_id
        pred2error[pred_id] = ErrorType.LOC
        matched = True
    return matched


def _match_pred_wrong_label_or_background(
        pred_idx: int,
        pred2error: Dict[int, str],
        pred2target: Dict[int, int],
        iou_matrix: np.array,
        im_targets_df: pd.DataFrame,
        im_preds_df: pd.DataFrame,
        iou_background: float,
        iou_foreground: float,
) -> None:
    """Try to match `pred_idx` to a target (with different label) and identify error

    If there is a match `pred2error` and  (maybe) `pred2target` are modified in place.
    Possible error types found in this function:
        ErrorType.BKG, ErrorType.CLS, ErrorType.CLS_LOC

    :param pred_idx: Index of prediction based on score (index 0 is maximum score for image).
    :param pred2error: Dict mapping pred_id to error type.
    :param pred2target: Dict mapping pred_id to target_id (if match found with iou above background)
    :param target2pred: Dict mapping target_id to pred_id to pred considered correct (if any).
    :param iou: Matrix with size [n_targets, n_preds] with IoU between all preds and targets.
    :param im_targets_df: DataFrame with targets for the image being processed.
    :param im_preds_df: DataFrame with preds for the image being processed.
    :param iou_background: Minimum IoU to consider a pred not background for target.
    :param iou_foreground: Minimum IoU to consider a pred foreground for a target.
    """
    # Find highest overlapping target for pred processed
    target_idx = np.argmax(iou_matrix[:, pred_idx])
    iou = np.max(iou_matrix[:, pred_idx])
    target_id = im_targets_df.at[target_idx, "target_id"]
    pred_id = im_preds_df.at[pred_idx, "pred_id"]

    if iou < iou_background:
        pred2error[pred_id] = ErrorType.BKG

    # preds with correct label do not get here. Thus, no need to check if label
    #   is wrong
    elif iou >= iou_foreground:
        pred2target[pred_id] = target_id
        pred2error[pred_id] = ErrorType.CLS
    else:
        # No match to target, as we cannot be sure model was remotely close to
        #   getting it right
        pred2error[pred_id] = ErrorType.CLS_LOC


def _find_missed_targets(
        im_targets_df: pd.DataFrame, pred2target: Dict[int, int]
) -> Set[int]:
    """Find targets in the processed image that were not matched by any prediction

    :param im_targets_df: DataFrame with targets for the image being processed.
    :param pred2target: Dict mapping pred_id to target_id (if match found with
        iou above background)
    :return missed_targets: Set of all the target ids that were missed

    """
    matched_targets = [t for t in pred2target.values() if t is not None]
    missed_targets = set(im_targets_df["target_id"]) - set(matched_targets)
    return missed_targets


def _format_errors_as_dataframe(
        pred2error: Dict[int, str],
        pred2target: Dict[int, int],
        missed_targets: Set[int],
) -> pd.DataFrame:
    """Use the variables used to classify errors to format them in a ready to use DataFrame

    :param pred2error: Dict mapping pred_id to error type.
    :param pred2target: Dict mapping pred_id to target_id (if match found with
        iou above background)
    :param missed_targets: Set of all the target ids that were missed
    :return: DataFrame with columns ERRORS_DF_COLUMNS
    """
    errors_df = pd.DataFrame.from_records(
        [
            {"pred_id": pred_id, "error_type": error}
            for pred_id, error in pred2error.items()
        ]
    )
    errors_df["target_id"] = None
    errors_df.set_index("pred_id", inplace=True)
    for pred_id, target_id in pred2target.items():
        errors_df.at[pred_id, "target_id"] = target_id

    missed_df = pd.DataFrame(
        {
            "pred_id": None,
            "error_type": ErrorType.MISS,
            "target_id": list(missed_targets),
        }
    )
    errors_df = pd.concat(
        [errors_df.reset_index(), missed_df], ignore_index=True
    ).astype(
        {"pred_id": pd.Int64Dtype(), "target_id": pd.Int64Dtype(), "error_type": pd.StringDtype()}
    )

    return errors_df

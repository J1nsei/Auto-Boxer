import pandas as pd
import argparse
from tqdm import tqdm
import json
from pathlib import Path
from typing import Tuple, Dict
import torch
import numpy as np
import ultralytics


def tuple_type(strings):
    strings = strings.replace("(", "").replace(")", "")
    mapped_int = map(int, strings.split(","))
    return tuple(mapped_int)


def main_arg_parser() -> argparse.Namespace:
    """
    Parse command line arguments.

    :return: argparse.Namespace containing the command line arguments.
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(description='Find model | dataset errors')
    parser.add_argument('--train', '--t', default='True', metavar='TRAIN_MODE', type=str, required=True,
                        help='Model mode. False == predict.')
    parser.add_argument('--data', '--d', metavar='DATA_PATH', type=str, default='dataset',
                        help='Dataset path')
    parser.add_argument('--model', '--m', metavar='MODEL', type=str, default='./models/default/yolov8s.pt',
                        help='YOLO model path')
    parser.add_argument('--create_dataset', default='False', type=str, help='Create YOLO dataset')
    parser.add_argument('--trainsz', default=0.7, type=float, help='Size of training partition')
    parser.add_argument('--impact', default='True', type=str, help='Model mode. False == predict.')
    parser.add_argument('--preds', default='', type=str, help='Use your predictions')
    parser.add_argument('--save_preds', default='False', type=str, help='Save your predictions')
    parser.add_argument('--vis', default='', type=str, help='Use saved visualization')
    parser.add_argument('--save_vdata', default='False', type=str, help='Save visualization dataset')
    args = parser.parse_args()
    return args


def get_utils_variables(data_path: Path) -> Tuple[Path, Dict[int, str]]:
    """
    Get utility variables from a data directory.

    :param data_path: The path to the data directory.
    :type data_path: Path
    :return: A tuple containing the images path (Path object) and a dictionary mapping category IDs to names.
    :rtype: Tuple[Path, Dict[int, str]]
    """
    images_path = data_path / "images"
    annotations_path = data_path / "labels.json"

    with open(annotations_path, "r") as f:
        targets_json = json.load(f)

    id2label = {cat["id"]: cat["name"] for cat in targets_json["categories"]}
    return images_path, id2label


def _uncenter_boxes(boxes_xywh: torch.Tensor) -> np.ndarray:
    """
    Takes a tensor `boxes_xywh` and converts it to a numpy array, removing centering.

    :param boxes_xywh: A tensor with bounding box data in the format [x_center, y_center, width, height]
    :type boxes_xywh: torch.Tensor
    :return: A numpy array with bounding box data in the format [x_top_left, y_top_left, width, height]
    :rtype: np.ndarray
    """
    boxes = boxes_xywh.cpu().numpy().copy()
    boxes[:, 0] -= boxes[:, 2] / 2.0
    boxes[:, 1] -= boxes[:, 3] / 2.0
    return boxes


def YOLOres2COCO(results: ultralytics.engine.results.Results, image_id: int) -> pd.DataFrame:
    """
    Convert YOLO detection results from Ultralytics to COCO-style format.

    :param results: An instance of Ultralytics Results containing detection results.
    :type results: ultralytics.engine.results.Results
    :param image_id: The ID of the image associated with the detection results.
    :type image_id: int
    :return: A DataFrame containing the converted COCO-style detection results.
    :rtype: pd.DataFrame
    """
    bboxes = _uncenter_boxes(results[0].boxes.xywh)
    scores = results[0].boxes.conf.cpu().numpy()
    categories = results[0].boxes.cls.cpu().numpy()
    res = pd.DataFrame(columns=['target_id', 'image_id', 'label_id', 'xmin', 'ymin', 'xmax', 'ymax', 'score'])
    res = res.astype({'target_id': 'int', 'image_id': 'int', 'label_id': 'int'})
    i = 0
    for catId in categories:
        bbox = bboxes[i].tolist()
        res.loc[i] = [i, image_id, catId, bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3], scores[i]]
        i += 1

    res = res.astype({'target_id': 'int', 'image_id': 'int', 'label_id': 'int'})
    return res


def get_predictions(
    model: ultralytics.models.yolo.model.YOLO,
    device,
    images_path: Path,
    images_df: pd.DataFrame,
    id2label: Dict,
    pred_args: Dict,
    PREDS: str = '',
    SAVE_PREDS: bool = False,
) -> pd.DataFrame:
    """
    Generate predictions using a YOLO model for a list of images.

    :param model: An instance of Ultralytics YOLO model.
    :type model: ultralytics.models.yolo.model.YOLO
    :param images_path: The path to the directory containing images.
    :type images_path: Path
    :param images_df: A DataFrame containing image metadata including file names and image IDs.
    :type images_df: pd.DataFrame
    :return: A DataFrame containing the predictions in COCO-style format.
    :rtype: pd.DataFrame
    """
    if PREDS:
        return pd.read_csv(PREDS)
    preds_df = pd.DataFrame(columns=["image_id", "label_id", "xmin", "ymin", "xmax", "ymax", "score"])
    for img in tqdm(range(len(images_df)), desc='Making predictions'):
        output = model.predict(images_path / images_df['file_name'][img], device=device, **pred_args)
        pred = YOLOres2COCO(output, images_df['image_id'][img]).drop(columns=['target_id'])
        preds_df = pd.concat([preds_df, pred], ignore_index=True)
    preds_df = preds_df.reset_index().rename(columns={"index": "pred_id"})
    yolo2coco = {i: cat for i, cat in enumerate(id2label)}
    preds_df['label_id'] = preds_df['label_id'].map(yolo2coco)
    preds_df.to_csv('preds', index=False) if SAVE_PREDS else None
    return preds_df

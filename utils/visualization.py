import fiftyone as fo
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Dict
import pandas as pd


def make_charts(impact: Dict[str, float]):
    """
    Create and display two charts based on the impact dictionary.

    :param impact: A dictionary containing impact values for different metrics.
    :type impact: dict
    """
    order = ["mAP@50", "classification", "localization", "cls & loc", "duplicate", "background", "missed"]
    labels = ["mAP@50\n base", "CLS", "LOC", "CLS & LOC", "DUP", "BKG", "MISS"]
    x = range(len(impact))
    y = [np.float64(impact[o]).clip(min=0) for o in order]
    colors = plt.cm.plasma(np.linspace(0, 1, len(labels)))

    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.bar(x, y, color=colors, tick_label=labels, zorder=10)
    plt.grid(alpha=0.5, zorder=1)
    plt.title("Impact of errors on the base metric")

    plt.subplot(122)
    plt.pie(y[1:], colors=colors[1:], labels=labels[1:], autopct='%1.1f%%')
    plt.title('Proportion of errors')
    plt.tight_layout()
    plt.savefig('results.png', bbox_inches='tight')
    plt.show()
    return


def create_fiftyone_dataset(data_path: Path, VIS: str = '') -> fo.Dataset:
    """
    Create a FiftyOne dataset from COCO-style data.

    :param data_path: The path to the directory containing COCO-style data.
    :type data_path: Path
    :return: A FiftyOne dataset.
    :rtype: fiftyone.core.dataset.Dataset
    """

    if VIS:
        print('Loading a FiftyOne dataset:')
        dataset = fo.Dataset.from_dir(
            overwrite=True,
            name='Error Analysis Dataset',
            dataset_dir=data_path,
            dataset_type=fo.types.COCODetectionDataset,
            data_path='./images/',
            labels_path=VIS,
            include_id=True,
            include_annotation_id=True
        )
    else:
        print('Creating a FiftyOne dataset:')
        dataset = fo.Dataset.from_dir(
            overwrite=True,
            name='Error Analysis Dataset',
            dataset_dir=data_path,
            dataset_type=fo.types.COCODetectionDataset,
            data_path='./images/',
            label_types='detections',
            include_id=True,
            include_annotation_id=True
        )
    return dataset


def convert_coordinates(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    image_width: float,
    image_height: float
) -> list:
    """
    Convert bounding box coordinates from absolute to normalized coordinates.

    :param xmin: The minimum x-coordinate of the bounding box.
    :type xmin: float
    :param ymin: The minimum y-coordinate of the bounding box.
    :type ymin: float
    :param xmax: The maximum x-coordinate of the bounding box.
    :type xmax: float
    :param ymax: The maximum y-coordinate of the bounding box.
    :type ymax: float
    :param image_width: The width of the image.
    :type image_width: float
    :param image_height: The height of the image.
    :type image_height: float
    :return: A list of normalized coordinates [normalized_x, normalized_y, normalized_width, normalized_height].
    :rtype: list
    """
    width = xmax - xmin
    height = ymax - ymin
    top_left_x = xmin
    top_left_y = ymin

    normalized_top_left_x = top_left_x / image_width
    normalized_top_left_y = top_left_y / image_height
    normalized_width = width / image_width
    normalized_height = height / image_height

    normalized_coordinates = [normalized_top_left_x, normalized_top_left_y, normalized_width, normalized_height]
    return normalized_coordinates


def convert_preds(
    dataset: fo.Dataset,
    preds_df: pd.DataFrame,
    errors_df: pd.DataFrame,
    id2label: Dict[int, str]
) -> None:
    """
    Convert predictions to FiftyOne format and attach them to samples in a FiftyOne dataset.

    :param dataset: The FiftyOne dataset to which the predictions will be attached.
    :type dataset: fiftyone.core.dataset.Dataset
    :param preds_df: DataFrame containing prediction data.
    :type preds_df: pd.DataFrame
    :param errors_df: DataFrame containing error data.
    :type errors_df: pd.DataFrame
    :param id2label: A dictionary mapping label IDs to label names.
    :type id2label: dict
    """
    for sample in tqdm(dataset, desc='Converting predictions to FiftyOne'):
        img_id = sample.coco_id
        image_width = sample.metadata['width']
        image_height = sample.metadata['height']

        # Convert predictions to FiftyOne format
        detections = []

        for _, row in preds_df.query('image_id==@img_id').iterrows():
            label = id2label[row['label_id']]
            confidence = row["score"]

            # Bounding box coordinates should be relative values
            # in [0, 1] in the following format:
            # [top-left-x, top-left-y, width, height]
            bounding_box = convert_coordinates(
                row['xmin'],
                row['ymin'],
                row['xmax'],
                row['ymax'],
                image_width,
                image_height
            )
            index = row['pred_id']
            tags = [errors_df.query('pred_id==@index')['error_type'].iloc[0]]
            detections.append(
                fo.Detection(
                    label=label,
                    bounding_box=bounding_box,
                    confidence=confidence,
                    index=index,
                    tags=tags
                )
            )

        # Store detections
        sample["predictions"] = fo.Detections(detections=detections)

        sample.save()
    return


def add_error_labels(
    dataset: fo.Dataset,
    missed_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    images_df: pd.DataFrame,
    images_path: Path
) -> None:
    """
    Add the 'missed' tag to detections in a FiftyOne dataset based on information from dataframes.

    :param dataset: The FiftyOne dataset to which the 'missed' tag will be added.
    :type dataset: fiftyone.core.dataset.Dataset
    :param missed_df: DataFrame containing missed target information.
    :type missed_df: pd.DataFrame
    :param targets_df: DataFrame containing target information.
    :type targets_df: pd.DataFrame
    :param images_df: DataFrame containing image information.
    :type images_df: pd.DataFrame
    :param images_path: The path to the directory containing image files.
    :type images_path: pathlib.Path
    """
    for target in tqdm(missed_df['target_id'], desc='Adding missed labels'):
        image_id = targets_df.query('target_id==@target')['image_id'].iloc[0]
        file_name = images_df.query('image_id==@image_id')['file_name'].iloc[0]
        file_path = os.path.abspath(images_path / file_name)
        sample = dataset[file_path]

        for detection in sample['detections']['detections']:
            if detection['coco_id'] == target:
                detection.tags.append("missed")
                sample.save()
                break
    return


def visualize(
    data_path: Path,
    images_path: Path,
    targets_df: pd.DataFrame,
    images_df: pd.DataFrame,
    preds_df: pd.DataFrame,
    errors_df: pd.DataFrame,
    id2label: Dict[int, str],
    VIS: str = '',
    SAVE_VDATA: bool = False
) -> None:
    """
    Visualize data using the FiftyOne app.

    :param data_path: The path to the data directory.
    :type data_path: pathlib.Path
    :param images_path: The path to the directory containing image files.
    :type images_path: pathlib.Path
    :param targets_df: DataFrame containing target information.
    :type targets_df: pd.DataFrame
    :param images_df: DataFrame containing image information.
    :type images_df: pd.DataFrame
    :param preds_df: DataFrame containing prediction data.
    :type preds_df: pd.DataFrame
    :param errors_df: DataFrame containing error data.
    :type errors_df: pd.DataFrame
    :param id2label: A dictionary mapping label IDs to label names.
    :type id2label: dict
    """

    # Create a FiftyOne dataset
    # if VIS:
    #     dataset = fo.load_dataset('./visualization')
    # else:
    #     dataset = create_fiftyone_dataset(data_path)
    dataset = create_fiftyone_dataset(data_path, VIS)
    # Convert predictions and add error labels
    if not VIS:
        convert_preds(dataset, preds_df, errors_df, id2label)
        missed_df = errors_df[errors_df["error_type"] == 'missed'].reset_index().drop(columns=['index'])
        add_error_labels(dataset, missed_df, targets_df, images_df, images_path)

    # if SAVE_VDATA:
        # print('Saving a FiftyOne dataset:\n')
        # dataset.export(
        #     export_dir = str(data_path / 'visualization'),
        #     dataset_type = fo.types.COCODetectionDataset,
        #     export_media = False,
        #     overwrite = True,
        #     label_field = ["detections", "predictions"]
        # )

    # Launch the FiftyOne app to visualize the data
    session = fo.launch_app(dataset)
    session.wait()

    # Print session information
    print(session)

    return

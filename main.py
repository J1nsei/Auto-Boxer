from ultralytics import YOLO
from utils.utils import *
from utils.dataset_creation import *
from utils.find_errors import *
from utils.visualization import *
from utils.evaluate_errors import *
import os
import yaml


def main():
    args = main_arg_parser()
    DATA_PATH = Path(args.data)
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data path '{DATA_PATH}' does not exist.")

    MODEL = args.model
    TRAIN_MODE = args.train.lower() == 'true'
    CREATE_DATASET = args.create_dataset.lower() == 'true'
    TRAIN_FRACTION = args.trainsz
    IMPACT = args.impact.lower() == 'true'
    PREDS = args.preds
    SAVE_PREDS = args.save_preds.lower() == 'true'
    VIS = args.vis
    SAVE_VDATA = args.save_vdata.lower() == 'true'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO(MODEL)

    if TRAIN_MODE:
        with open('models/train_config.yaml', 'r') as train_cfg:
            train_args = yaml.safe_load(train_cfg)
        train_args['imgsz'] = tuple(train_args['imgsz'])
        if train_args['imgsz'][0] == train_args['imgsz'][1] == 0:
            raise ValueError('Insert image sizes in train_config.yaml')
        create_yolo_dataset(DATA_PATH, TRAIN_FRACTION) if CREATE_DATASET else None
        model.train(device=device, **train_args)
    else:
        with open('models/pred_config.yaml', 'r') as pred_cfg:
            pred_args = yaml.safe_load(pred_cfg)
        images_df, targets_df = load_dataset(DATA_PATH)
        images_path, id2label = get_utils_variables(DATA_PATH)
        preds_df = get_predictions(model, device, images_path, images_df, id2label, pred_args, PREDS, SAVE_PREDS)
        errors_df = classify_predictions_errors(targets_df, preds_df)
        print(errors_df["error_type"].value_counts())
        if IMPACT:
            impact = calculate_error_impact(
                "mAP@50",
                MyMeanAveragePrecision(foreground_threshold=FOREGROUND_IOU_THRESHOLD),
                errors_df,
                targets_df,
                preds_df
            )
            make_charts(impact)
        visualize(DATA_PATH, images_path, targets_df, images_df, preds_df, errors_df, id2label, VIS, SAVE_VDATA)
    print('\nDONE')


if __name__ == "__main__":
    main()

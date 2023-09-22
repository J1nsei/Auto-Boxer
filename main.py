from ultralytics import YOLO
from utils.utils import *
from utils.dataset_creation import *
from utils.find_errors import *
from utils.visualization import *
from utils.evaluate_errors import *
import os


def main():
    args = main_arg_parser()
    DATA_PATH = Path(args.data)
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data path '{DATA_PATH}' does not exist.")

    MODEL = args.model
    TRAIN_MODE = args.train.lower() == 'true'
    TRAIN_FRACTION = args.trainsz
    IMGSZ = args.imgsz
    IMPACT = args.impact.lower() == 'true'
    PREDS = args.preds
    SAVE_PREDS = args.save_preds.lower() == 'true'
    VIS = args.vis
    SAVE_VDATA = args.save_vdata.lower() == 'true'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = YOLO(MODEL)
    if TRAIN_MODE:
        create_yolo_dataset(DATA_PATH, TRAIN_FRACTION)
        model.train(data='data.yaml', imgsz=IMGSZ, epochs=20, batch=-1, save=True, verbose=True, seed=0)
    else:
        images_df, targets_df = load_dataset(DATA_PATH)
        images_path, id2label = get_utils_variables(DATA_PATH)
        preds_df = get_predictions(model, device, images_path, images_df, id2label, PREDS, SAVE_PREDS)
        errors_df = classify_predictions_errors(targets_df, preds_df)
        print(f"Total predictions: {len(preds_df)}, Total errors: {len(errors_df)}")
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
            print('charts')
        visualize(DATA_PATH, images_path, targets_df, images_df, preds_df, errors_df, id2label, VIS, SAVE_VDATA)
    print('\nDONE')


if __name__ == "__main__":
    main()

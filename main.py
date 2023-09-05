from ultralytics import YOLO
from utils.utils import *
from utils.dataset_creation import *
from utils.find_errors import *
from utils.visualization import *
from utils.evaluate_errors import *


def main():
    parser = argparse.ArgumentParser(description="Your description here")
    parser.add_argument('--data', type=str, required=True, help='Path to your data')
    parser.add_argument('--model', type=str, required=True, help='Your YOLO model name')
    parser.add_argument('--train', type=str, required=True, help='Training mode (true/false)')
    parser.add_argument('--trainsz', type=float, default=0.8, help='Training data fraction')
    parser.add_argument('--imgsz', type=int, default=416, help='Image size')
    parser.add_argument('--impact', type=bool, default=False, help='Calculate error impact (true/false)')

    args = parser.parse_args()

    DATA_PATH = Path(args.data)
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data path '{DATA_PATH}' does not exist.")

    MODEL = args.model
    TRAIN_MODE = args.train.lower() == 'true'
    TRAIN_FRACTION = args.trainsz
    IMGSZ = args.imgsz

    model = YOLO(MODEL)
    IMPACT = args.impact

    if TRAIN_MODE:
        create_yolo_dataset(DATA_PATH, TRAIN_FRACTION)
        model.train(data='data.yaml', imgsz=IMGSZ, epochs=20, batch=-1, save=True, verbose=True, seed=0)
    else:
        images_df, targets_df = load_dataset(DATA_PATH)
        images_path, id2label = get_utils_variables(DATA_PATH)
        preds_df = get_predictions(model, images_path, images_df)
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
        visualize(DATA_PATH, images_path, targets_df, images_df, preds_df, errors_df, id2label)
    print('\nDONE')


if __name__ == "__main__":
    main()

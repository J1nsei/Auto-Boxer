# Auto-Boxer

![](https://github.com/J1nsei/Auto-Boxer/blob/main/utils/idea.gif)

## Description
**Auto-Boxer** is a tool designed to facilitate and automate the process of analyzing data annotation in the context of object detection. It provides powerful capabilities to detect and display errors in data annotation, such as misplaced class labels, incorrectly defined bounding boxes, or even missing annotation. In addition, there is a useful visualization of errors using *FiftyOne*, as well as analysis of their contribution to the *mAP* metric. A small demonstration of the program results is available in [demo.ipynb](https://github.com/J1nsei/Auto-Boxer/blob/main/demo.ipynb).

### Error types:
- **Classification error (CLS)**: incorrect class label (i.e. localized correctly but classified incorrectly).
- **Localization error (LOC)**: localized incorrectly, classified correctly. 
- **Classification and Localization error (CLS & LOC)**: incorrect classification and localization at the same time. 
- **Duplicate detection error (DUP)**: Target detected correctly, but there is a more correct detection (confidence score higher).
- **Background error (BKG)**: Background detected as foreground.
- **Missed target error (MISS)**: Undetected targets that have not been marked as classification or localization errors.


## Installation
1. Install Python >= 3.10.
2. Install [PyTorch](https://pytorch.org/get-started/locally/) (GPU recommended).
3. Install dependencies:
```python -m pip install -r requirements.txt```

## Usage
- Run in error search mode (for training *train=True*):
    - Linux:
        ``` python3 main.py --data='./dataset' --model='./models/default/demo.pt' --train=False --impact=True ```

    - Windows:
        ``` python main.py --data=".\\dataset" --model=".\\models\\default\\demo.pt" --train=False --impact=True ```
- Description of parameters:
    - **data** = './dataset' (path to the data directory).
    - **model** = './models/my_model.pt' (model)
    - **impact** = True (enabling calculation of the impact of errors on the metric)
## Data format and storage
- Data must be labeled in **COCO** format.
- Storage Format:
    ```bash
    ├── dataset
    │   ├── labels.json  <--------- annotation file
    │   └── images       <--------- images folder
    │       ├── img1.jpg
    │       ├── ...
    │       └── imgN.jpg
    ├── models
    ├── utils
    ├── main.py
    └── README.md
    ```
## Using your own model
To use your own trained YOLOv8 model, place the file with the weights in the *models* folder and specify the path to them when running (```--model='./models/my_model.pt'``` ).

## Task list
- [ ]  Add saving for error labeled fiftyone dataset.
- [x]  Add a parameter dictionary reading for the YOLO model.
- [ ]  Complete the contents of the readme file. 
- [x]  Add a demo.

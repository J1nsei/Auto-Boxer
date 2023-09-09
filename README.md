# Auto-Boxer

![](https://github.com/J1nsei/Auto-Boxer/blob/main/utils/idea.gif)

## Описание
**Auto-Boxer** - это инструмент, разработанный для облегчения и автоматизации процесса анализа разметки данных в задачах компьютерного зрения и машинного обучения. Он предоставляет мощные возможности по обнаружению и отображению ошибок в разметке данных, такие как неправильно установленные метки классов, некорректно заданные ограничительные рамки (bounding boxes) или даже отсутствие разметки. Кроме того, присутствует удобная визуализация ошибок с помощью *fiftyone*, а также анализ их вклада в метрику *mAP*. 

### Типы ошибок:
- **Classification error (CLS)**: некорректная метка класса (т.е локализовано верно, но классифицировано неверно).
- **Localization error (LOC)**: локализовано неправильно, классифицировано верно. 
- **Classification and Localization error (CLS & LOC)**: неверная классификация и локализация одновременно. 
- **Duplicate detection error (DUP)**: цель обнаружена верно, однако существует более правильная детекция (confidence score выше).
- **Background error (BKG)**: обнаружен фон как передний план.
- **Missed target error (MISS)**: необнаруженные цели, которые не были отмечены как ошибки классификации или локализации.


## Установка
1. Установите Python >= 3.10.
2. Установите [PyTorch](https://pytorch.org/get-started/locally/) (рекомендуется использовать GPU).
3. Установите зависимости:
```python -m pip install -r requirements.txt```

## Использование
- Запуск в режиме поиска ошибок (для обучения *train=True*):
  Linux:
``` python3 main.py --data='./dataset' --model='./models/default/demo.pt' --train=False --impact=True ```
  Windows:
``` python3 main.py --data=".\\dataset" --model=".\\models\\default\\demo.pt" --train=False --impact=True ```
- Описание параметров:
    - **data** = './dataset' (путь к директории с данными).
    - **model** = './models/my_model.pt' (модель)
    - **impact** = True (включение рассчета влияния ошибок на метрику)
## Формат и хранение данных
- Данные должны быть размечены в формате **COCO**.
- **Метки** классов должны быть **непрерывными и начинаться с 0**. 
- Формат хранения:
    ```bash
    ├── dataset
    │   ├── labels.json  <--------- файл с аннотациями
    │   └── images       <--------- изображения
    │       ├── img1.jpg
    │       ├── ...
    │       └── imgN.jpg
    ├── models
    ├── utils
    ├── main.py
    └── README.md
    ```
## Использование собственной модели
Чтобы использовать собственную натренированную модель YOLOv8, поместите файл с весами в папку *models* и при запуске укажите к ним путь (```--model='./models/my_model.pt'``` ).

## Task list
- [ ]  Add saving for error labeled fiftyone dataset.
- [ ]  Add a parameter dictionary reading for the YOLO model.
- [ ]  Complete the contents of the readme file. 
- [x]  Add a demo.

# Supervisely app: car perspective classification

[![Project](https://img.shields.io/badge/Project-AutoInspect-black?logo=github)](https://github.com/DedovInside/AutoInspect/tree/ml/ml)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-car--view-yellow?logo=huggingface)](https://huggingface.co/mitbersh/car-view)

![logo-ml.png](https://raw.githubusercontent.com/brshtsk/SuperviselyPerspective/main/img/logo-ml.png)

Папка содержит полноценный каркас Supervisely app и CLI-скрипт для отладки.

## Что внутри

- `config.json` — манифест app.
- `src/app.py` — UI app (single image inference + массовая разметка датасета тегами).
- `src/main.py` — backend-логика инференса (локальный путь или `image_id`).
- `src/model_store.py` — загрузка и кеш модели из Hugging Face.

Модель по умолчанию:

`https://huggingface.co/mitbersh/car-view/resolve/main/car_view_model.pth`

## Запуск как Supervisely app

1. Для проверки на одном изображении укажите `Image ID` и нажмите **Run inference**.
2. Для массовой разметки укажите `Dataset ID`, `Tag name` и нажмите **Tag dataset**.

`Tag dataset` создаёт image tag (если его нет в meta проекта) и записывает в него `predicted_class` для каждого изображения датасета.

## Демо-скриншоты

### Массовая разметка датасета

![dataset-tag.png](https://raw.githubusercontent.com/brshtsk/SuperviselyPerspective/main/img/dataset-tag.png)

### Инференс изображения

![image-infer.png](https://raw.githubusercontent.com/brshtsk/SuperviselyPerspective/main/img/image-infer.png)

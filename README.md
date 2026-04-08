# Supervisely app: car perspective classification

![logo-ml.png](img/logo-ml.png)

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


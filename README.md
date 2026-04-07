# Supervisely app: car perspective classification

Папка содержит полноценный каркас Supervisely app и CLI-скрипт для отладки.

## Что внутри

- `config.json` — манифест app.
- `src/app.py` — UI app (ввод `image_id`, запуск инференса, вывод результата).
- `src/main.py` — backend-логика инференса (локальный путь или `image_id`).
- `src/model_store.py` — загрузка и кеш модели из Hugging Face.

Модель по умолчанию:

`https://huggingface.co/mitbersh/car-view/resolve/main/car_view_model.pth`

## Установка

```bash
pip install -r requirements.txt
```

## Локальная проверка (CLI)

```bash
python src/main.py --image "C:/path/to/image.jpg"
python src/main.py --image-id 12345
```

## Запуск как Supervisely app

1. Импортируйте папку `supervisely` как app в ваш Supervisely instance.
2. Убедитесь, что в окружении app доступны `SERVER_ADDRESS` и `API_TOKEN`.
3. Запустите app, в UI укажите `image_id` и нажмите **Run inference**.

Если нужно, можно переопределить URL модели через аргументы в `src/main.py` или через доработку `src/app.py`.


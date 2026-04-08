import os
from typing import Any, List

import supervisely as sly
import uvicorn
from supervisely.app.widgets import Button, Card, Container, Field, Input, InputNumber, Text

try:
    from supervisely.app.widgets import Progress
except Exception:
    Progress = None

from main import predict_supervisely_image_id, tag_supervisely_dataset


FIXED_IMAGE_SIZE = 224

image_id_input = Input(value="", placeholder="e.g. 12345")
dataset_id_input = Input(value="", placeholder="e.g. 67890")
top_k_input = InputNumber(min=1, max=9, step=1, value=3)
device_input = Input(value="auto", placeholder="auto | cpu | cuda | mps")
tag_name_input = Input(value="car_view", placeholder="Image tag name")

run_button = Button("Run inference", button_type="primary")
tag_dataset_button = Button("Tag dataset", button_type="success")

status_text = Text("", status="text")
predicted_class_text = Text("", status="info")
confidence_text = Text("", status="info")
top_k_text = Text("", status="text")
dataset_progress_text = Text("", status="text")
dataset_eta_text = Text("", status="text")
dataset_progress = Progress() if Progress is not None else None

result_widgets: List[Any] = [
    status_text,
    predicted_class_text,
    confidence_text,
    top_k_text,
    dataset_progress_text,
    dataset_eta_text,
]
if dataset_progress is not None:
    result_widgets.append(dataset_progress)

controls = Card(
    title="Car view classification",
    content=Container(
        [
            Field(content=image_id_input, title="Image ID"),
            Field(content=dataset_id_input, title="Dataset ID"),
            Field(content=top_k_input, title="Top-k"),
            Field(content=Text(f"{FIXED_IMAGE_SIZE}", status="text"), title="Image size (fixed)"),
            Field(content=device_input, title="Device"),
            Field(content=tag_name_input, title="Tag name"),
            run_button,
            tag_dataset_button,
        ]
    ),
)

results = Card(
    title="Result",
    content=Container(result_widgets),
)

layout = Container([controls, results])
app = sly.Application(layout=layout)


def _format_top_k(top_items):
    lines = []
    for rank, item in enumerate(top_items, start=1):
        cls = item.get("class", "unknown")
        conf = float(item.get("confidence", 0.0))
        lines.append(f"{rank}. {cls}: {conf:.2%}")
    return "\n".join(lines)


def _parse_positive_int(value, field_name: str) -> int:
    text = str(value).strip()
    if text == "":
        raise ValueError(f"{field_name} is required")
    parsed = int(text)
    if parsed < 1:
        raise ValueError(f"{field_name} must be >= 1")
    return parsed


def _format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, sec = divmod(rem, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {sec}s"
    if minutes > 0:
        return f"{minutes}m {sec}s"
    return f"{sec}s"


def _set_progress(current: int, total: int) -> None:
    if dataset_progress is None:
        return

    try:
        if hasattr(dataset_progress, "set_total"):
            dataset_progress.set_total(total)

        if hasattr(dataset_progress, "set_current_value"):
            dataset_progress.set_current_value(current)
        elif hasattr(dataset_progress, "set_current"):
            dataset_progress.set_current(current)
        elif hasattr(dataset_progress, "update"):
            dataset_progress.update(current)
    except Exception:
        sly.logger.debug("Progress widget update skipped due to widget API mismatch")


@run_button.click
def run_inference() -> None:
    try:
        image_id = _parse_positive_int(image_id_input.get_value(), "Image ID")
    except Exception as exc:
        status_text.set(f"Error: {exc}", status="error")
        return

    top_k = int(top_k_input.get_value())
    img_size = FIXED_IMAGE_SIZE
    device = str(device_input.get_value()).strip() or "auto"

    status_text.set("Running inference...", status="text")
    predicted_class_text.set("", status="info")
    confidence_text.set("", status="info")
    top_k_text.set("", status="text")
    dataset_progress_text.set("", status="text")
    dataset_eta_text.set("", status="text")
    _set_progress(0, 1)

    try:
        result = predict_supervisely_image_id(
            image_id=image_id,
            top_k=top_k,
            img_size=img_size,
            device_arg=device,
        )
    except Exception as exc:
        sly.logger.exception("Inference failed")
        status_text.set(f"Error: {exc}", status="error")
        return

    predicted_class = result["predicted_class"]
    confidence = float(result["confidence"])
    formatted_top = _format_top_k(result.get("top_k", []))

    status_text.set("Inference completed", status="success")
    predicted_class_text.set(f"Predicted class: {predicted_class}", status="info")
    confidence_text.set(f"Confidence: {confidence:.2%}", status="info")
    top_k_text.set(f"Top-k:\n{formatted_top}", status="text")


@tag_dataset_button.click
def run_dataset_tagging() -> None:
    try:
        dataset_id = _parse_positive_int(dataset_id_input.get_value(), "Dataset ID")
    except Exception as exc:
        status_text.set(f"Error: {exc}", status="error")
        return

    top_k = int(top_k_input.get_value())
    img_size = FIXED_IMAGE_SIZE
    device = str(device_input.get_value()).strip() or "auto"
    tag_name = str(tag_name_input.get_value()).strip() or "car_view"

    status_text.set("Tagging dataset...", status="text")
    predicted_class_text.set("", status="info")
    confidence_text.set("", status="info")
    top_k_text.set("", status="text")
    dataset_progress_text.set("Preparing...", status="text")
    dataset_eta_text.set("", status="text")
    _set_progress(0, 1)

    def _on_progress(progress_data):
        processed = int(progress_data.get("processed", 0))
        total = int(progress_data.get("total", 0))
        success = int(progress_data.get("success", 0))
        failed = int(progress_data.get("failed", 0))
        eta = float(progress_data.get("eta_seconds", 0.0))
        avg = float(progress_data.get("avg_seconds_per_image", 0.0))

        _set_progress(processed, max(total, 1))
        dataset_progress_text.set(
            f"Progress: {processed}/{total} | success: {success} | failed: {failed}", status="text"
        )
        dataset_eta_text.set(f"ETA: {_format_duration(eta)} | avg: {avg:.2f} s/img", status="text")

    try:
        stats = tag_supervisely_dataset(
            dataset_id=dataset_id,
            tag_name=tag_name,
            overwrite=True,
            top_k=top_k,
            img_size=img_size,
            device_arg=device,
            progress_cb=_on_progress,
        )
    except Exception as exc:
        sly.logger.exception("Dataset tagging failed")
        status_text.set(f"Error: {exc}", status="error")
        return

    status_text.set("Dataset tagging completed", status="success")
    predicted_class_text.set(
        f"Total: {stats['total']}, success: {stats['success']}, failed: {stats['failed']}",
        status="info",
    )
    dataset_eta_text.set(
        f"Done in {_format_duration(float(stats.get('elapsed_seconds', 0.0)))} | avg: {float(stats.get('avg_seconds_per_image', 0.0)):.2f} s/img",
        status="text",
    )
    _set_progress(int(stats.get("total", 0)), max(int(stats.get("total", 0)), 1))


if __name__ == "__main__":
    sly.logger.info("Starting Supervisely app: Car View Classifier")
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app.get_server(), host=host, port=port, log_level="info")

import supervisely as sly
from supervisely.app.widgets import Button, Card, Container, Field, Input, InputNumber, Text

from main import predict_supervisely_image_id, tag_supervisely_dataset


image_id_input = InputNumber(min=1, step=1, value=1)
dataset_id_input = InputNumber(min=1, step=1, value=1)
top_k_input = InputNumber(min=1, max=9, step=1, value=3)
img_size_input = InputNumber(min=32, max=1024, step=32, value=224)
device_input = Input(value="auto", placeholder="auto | cpu | cuda | mps")
tag_name_input = Input(value="car_view", placeholder="Image tag name")

run_button = Button("Run inference", button_type="primary")
tag_dataset_button = Button("Tag dataset", button_type="success")

status_text = Text("", status="text")
predicted_class_text = Text("", status="info")
confidence_text = Text("", status="info")
top_k_text = Text("", status="text")

controls = Card(
    title="Car view classification",
    content=Container(
        [
            Field(content=image_id_input, title="Image ID"),
            Field(content=dataset_id_input, title="Dataset ID"),
            Field(content=top_k_input, title="Top-k"),
            Field(content=img_size_input, title="Image size"),
            Field(content=device_input, title="Device"),
            Field(content=tag_name_input, title="Tag name"),
            run_button,
            tag_dataset_button,
        ]
    ),
)

results = Card(
    title="Result",
    content=Container([status_text, predicted_class_text, confidence_text, top_k_text]),
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


@run_button.click
def run_inference() -> None:
    image_id = int(image_id_input.get_value())
    top_k = int(top_k_input.get_value())
    img_size = int(img_size_input.get_value())
    device = str(device_input.get_value()).strip() or "auto"

    status_text.set("Running inference...", status="text")
    predicted_class_text.set("", status="info")
    confidence_text.set("", status="info")
    top_k_text.set("", status="text")

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
    dataset_id = int(dataset_id_input.get_value())
    top_k = int(top_k_input.get_value())
    img_size = int(img_size_input.get_value())
    device = str(device_input.get_value()).strip() or "auto"
    tag_name = str(tag_name_input.get_value()).strip() or "car_view"

    status_text.set("Tagging dataset...", status="text")
    predicted_class_text.set("", status="info")
    confidence_text.set("", status="info")
    top_k_text.set("", status="text")

    try:
        stats = tag_supervisely_dataset(
            dataset_id=dataset_id,
            tag_name=tag_name,
            overwrite=True,
            top_k=top_k,
            img_size=img_size,
            device_arg=device,
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


if __name__ == "__main__":
    sly.logger.info("Starting Supervisely app: Car View Classifier")

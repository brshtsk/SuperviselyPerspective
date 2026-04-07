import supervisely as sly
from supervisely.app.widgets import Button, Card, Container, Field, Input, InputNumber, Text

from main import predict_supervisely_image_id


image_id_input = InputNumber(min=1, step=1, value=1)
top_k_input = InputNumber(min=1, max=9, step=1, value=3)
img_size_input = InputNumber(min=32, max=1024, step=32, value=224)
device_input = Input(value="auto", placeholder="auto | cpu | cuda | mps")

run_button = Button("Run inference", button_type="primary")

status_text = Text("", status="text")
predicted_class_text = Text("", status="info")
confidence_text = Text("", status="info")
top_k_text = Text("", status="text")

controls = Card(
    title="Car view classification",
    content=Container(
        [
            Field(content=image_id_input, title="Image ID"),
            Field(content=top_k_input, title="Top-k"),
            Field(content=img_size_input, title="Image size"),
            Field(content=device_input, title="Device"),
            run_button,
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

    status_text.set("Running inference...")
    predicted_class_text.set("")
    confidence_text.set("")
    top_k_text.set("")

    try:
        result = predict_supervisely_image_id(
            image_id=image_id,
            top_k=top_k,
            img_size=img_size,
            device_arg=device,
        )
    except Exception as exc:
        sly.logger.exception("Inference failed")
        status_text.set(f"Error: {exc}")
        return

    predicted_class = result["predicted_class"]
    confidence = float(result["confidence"])
    formatted_top = _format_top_k(result.get("top_k", []))

    status_text.set("Inference completed")
    predicted_class_text.set(f"Predicted class: {predicted_class}")
    confidence_text.set(f"Confidence: {confidence:.2%}")
    top_k_text.set(f"Top-k:\n{formatted_top}")


if __name__ == "__main__":
    sly.logger.info("Starting Supervisely app: Car View Classifier")

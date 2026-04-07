import argparse
import importlib.util
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from model_store import DEFAULT_MODEL_URL, ensure_model_downloaded


def _load_infer_module():
    module_path = Path(__file__).resolve().parent / "infer_perspective.py"
    if not module_path.exists():
        raise RuntimeError(f"Infer module file not found: {module_path}")

    spec = importlib.util.spec_from_file_location("infer_perspective", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load infer module from: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_predictor(
    model_url: str = DEFAULT_MODEL_URL,
    device_arg: str = "auto",
    img_size: int = 224,
    top_k: int = 3,
):
    infer = _load_infer_module()
    model_path = ensure_model_downloaded(model_url=model_url)
    device = infer.get_device(device_arg)
    model = infer.load_model(model_path=model_path, device=device)

    def predict_image_path(image_path: str) -> Dict[str, Any]:
        return infer.predict_from_image_path(
            image_path=image_path,
            model=model,
            img_size=img_size,
            device=device,
            top_k=top_k,
        )

    return predict_image_path


def predict_single_file(
    image_path: str,
    model_url: str = DEFAULT_MODEL_URL,
    device_arg: str = "auto",
    img_size: int = 224,
    top_k: int = 3,
) -> Dict[str, Any]:
    predictor = build_predictor(
        model_url=model_url,
        device_arg=device_arg,
        img_size=img_size,
        top_k=top_k,
    )
    return predictor(image_path)


def predict_supervisely_image_id(
    image_id: int,
    model_url: str = DEFAULT_MODEL_URL,
    device_arg: str = "auto",
    img_size: int = 224,
    top_k: int = 3,
    work_dir: Optional[str] = None,
) -> Dict[str, Any]:
    import supervisely as sly

    api = sly.Api.from_env()
    image_info = api.image.get_info_by_id(image_id)
    if image_info is None:
        raise ValueError(f"Image not found in Supervisely by id: {image_id}")

    predictor = build_predictor(
        model_url=model_url,
        device_arg=device_arg,
        img_size=img_size,
        top_k=top_k,
    )

    local_dir = work_dir or tempfile.mkdtemp(prefix="sly_car_view_")
    os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, f"{image_id}_{image_info.name}")

    api.image.download_path(image_id, local_path)
    result = predictor(local_path)

    return {
        "image_id": image_id,
        "image_name": image_info.name,
        **result,
    }


def tag_supervisely_dataset(
    dataset_id: int,
    tag_name: str = "car_view",
    overwrite: bool = True,
    model_url: str = DEFAULT_MODEL_URL,
    device_arg: str = "auto",
    img_size: int = 224,
    top_k: int = 3,
) -> Dict[str, Any]:
    import supervisely as sly

    infer = _load_infer_module()
    class_names = list(getattr(infer, "CLASS_NAMES", []))
    if len(class_names) == 0:
        raise RuntimeError("CLASS_NAMES not found in infer_perspective.py")

    api = sly.Api.from_env()
    dataset_info = api.dataset.get_info_by_id(dataset_id)
    if dataset_info is None:
        raise ValueError(f"Dataset not found by id: {dataset_id}")

    project_meta = sly.ProjectMeta.from_json(api.project.get_meta(dataset_info.project_id))
    tag_meta = project_meta.get_tag_meta(tag_name)

    if tag_meta is None:
        tag_meta = sly.TagMeta(
            name=tag_name,
            value_type=sly.TagValueType.ONEOF_STRING,
            possible_values=class_names,
        )
        project_meta = project_meta.add_tag_meta(tag_meta)
        api.project.update_meta(dataset_info.project_id, project_meta.to_json())

    images = api.image.get_list(dataset_id)
    predictor = build_predictor(
        model_url=model_url,
        device_arg=device_arg,
        img_size=img_size,
        top_k=top_k,
    )

    total = len(images)
    success = 0
    failed = 0

    with tempfile.TemporaryDirectory(prefix="sly_car_view_batch_") as tmp_dir:
        for image_info in images:
            try:
                local_path = os.path.join(tmp_dir, f"{image_info.id}_{image_info.name}")
                api.image.download_path(image_info.id, local_path)

                result = predictor(local_path)
                predicted_class = result["predicted_class"]

                ann_json = api.annotation.download_json(image_info.id)
                ann = sly.Annotation.from_json(ann_json, project_meta)

                tags = list(ann.img_tags)
                if overwrite:
                    tags = [tag for tag in tags if tag.meta.name != tag_name]

                tags.append(sly.Tag(meta=tag_meta, value=predicted_class))
                new_ann = ann.clone(img_tags=sly.TagCollection(tags))
                api.annotation.upload_ann(image_info.id, new_ann)
                success += 1
            except Exception:
                sly.logger.exception(f"Failed to tag image id={image_info.id}")
                failed += 1

    return {
        "dataset_id": dataset_id,
        "tag_name": tag_name,
        "total": total,
        "success": success,
        "failed": failed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Supervisely helper for car perspective inference")
    parser.add_argument("--image", help="Path to local image")
    parser.add_argument("--image-id", type=int, help="Supervisely image id")
    parser.add_argument("--model-url", default=DEFAULT_MODEL_URL, help="Model URL (.pth)")
    parser.add_argument("--img-size", type=int, default=224, help="Model input size")
    parser.add_argument("--top-k", type=int, default=3, help="Top-k predictions")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Torch device",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if bool(args.image) == bool(args.image_id):
        raise ValueError("Use exactly one of --image or --image-id")

    if args.image:
        result = predict_single_file(
            image_path=args.image,
            model_url=args.model_url,
            device_arg=args.device,
            img_size=args.img_size,
            top_k=args.top_k,
        )
    else:
        result = predict_supervisely_image_id(
            image_id=args.image_id,
            model_url=args.model_url,
            device_arg=args.device,
            img_size=args.img_size,
            top_k=args.top_k,
        )

    print(f"Predicted class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.2%}")
    if args.top_k > 1:
        print("Top-k:")
        for rank, item in enumerate(result["top_k"], start=1):
            print(f"  {rank}. {item['class']}: {item['confidence']:.2%}")


if __name__ == "__main__":
    main()


"""
Take an ImageNet class number (0-999, per classes.txt) as input, run every
validation image of that class (from
/mnt/data/Public_datasets/imagenet/imagenet_pytorch/val/) through CLIP zero-
shot classification against all 1000 classes.txt labels, and check whether
CLIP predicts the same (correct) label for all of them.
"""

import os

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "openai/clip-vit-base-patch32"
CLASSES_PATH = "mod_classes2.txt"
IMAGENET_VAL_DIR = "/mnt/data/Public_datasets/imagenet/imagenet_pytorch/val"


def load_classes(path: str = CLASSES_PATH) -> list[str]:
    classes = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            _, label = line.split(":", 1)
            classes.append(label.strip())
    return classes


def load_wnids(val_dir: str = IMAGENET_VAL_DIR) -> list[str]:
    return sorted(os.listdir(val_dir))


def images_for_class(class_number: int, wnids: list[str], val_dir: str = IMAGENET_VAL_DIR) -> list[str]:
    wnid = wnids[class_number]
    class_dir = os.path.join(val_dir, wnid)
    return [os.path.join(class_dir, name) for name in sorted(os.listdir(class_dir))]


def load_clip(model_name: str = MODEL_NAME, device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor, device


def predict_labels(image_paths, classes, model, processor, device):
    prompts = [f"a photo of a {label}" for label in classes]
    images = [Image.open(path).convert("RGB") for path in image_paths]
    inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)
    top_indices = probs.argmax(dim=1)
    return [classes[i] for i in top_indices.tolist()]


def ask_class_number(classes: list[str]) -> int:
    while True:
        raw = input(f"Class number (0-{len(classes) - 1}): ").strip()
        if not raw.isdigit() or not (0 <= int(raw) < len(classes)):
            print(f"Please enter a number between 0 and {len(classes) - 1}.")
            continue
        return int(raw)


if __name__ == "__main__":
    classes = load_classes()
    wnids = load_wnids()
    model, processor, device = load_clip()

    print("Press Ctrl+C to quit.\n")
    try:
        while True:
            class_number = ask_class_number(classes)
            true_label = classes[class_number]
            image_paths = images_for_class(class_number, wnids)

            predicted_labels = predict_labels(image_paths, classes, model, processor, device)
            mismatches = [
                (path, label)
                for path, label in zip(image_paths, predicted_labels)
                if label != true_label
            ]

            print(f"\nClass {class_number}: {true_label} ({len(image_paths)} images)")
            if not mismatches:
                print(f"All images were classified as '{true_label}'.")
            else:
                print(f"{len(mismatches)} of {len(image_paths)} images were NOT classified as '{true_label}':")
                for path, label in mismatches:
                    print(f"  {path}: {label}")
            print("\n" + "-" * 40 + "\n")
    except KeyboardInterrupt:
        print("\nBye!")

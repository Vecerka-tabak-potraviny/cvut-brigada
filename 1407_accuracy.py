"""
Take a starting class number (0-989, per classes.txt) as input, take that
class and the next 9 (10 classes total), run every validation image of
those classes (from
/mnt/data/Public_datasets/imagenet/imagenet_pytorch/val/) through CLIP
zero-shot classification against all 1000 classes.txt labels, and report
the combined Top-1 accuracy across all 10 classes.
"""

import os

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "openai/clip-vit-base-patch32"
CLASSES_PATH = "mod_classes2.txt"
IMAGENET_VAL_DIR = "/mnt/data/Public_datasets/imagenet/imagenet_pytorch/val"
NUM_CLASSES = 10


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


def ask_start_number(classes: list[str], num_classes: int = NUM_CLASSES) -> int:
    max_start = len(classes) - num_classes - 1
    while True:
        raw = input(f"Starting class number (0-{max_start}): ").strip()
        if not raw.isdigit() or not (0 <= int(raw) <= max_start):
            print(f"Please enter a number between 0 and {max_start}.")
            continue
        return int(raw)


if __name__ == "__main__":
    classes = load_classes()
    wnids = load_wnids()
    model, processor, device = load_clip()

    print("Press Ctrl+C to quit.\n")
    try:
        while True:
            start = ask_start_number(classes)
            class_numbers = range(start, start + NUM_CLASSES)

            correct = 0
            total = 0
            print(f"\nClasses {start}-{start + NUM_CLASSES - 1}:")
            for class_number in class_numbers:
                true_label = classes[class_number]
                image_paths = images_for_class(class_number, wnids)
                predicted_labels = predict_labels(image_paths, classes, model, processor, device)

                class_correct = sum(1 for label in predicted_labels if label == true_label)
                correct += class_correct
                total += len(image_paths)
                print(f"  {class_number}: {true_label}: {class_correct}/{len(image_paths)} correct")

            accuracy = correct / total
            print(f"\nTop-1 accuracy over these {NUM_CLASSES} classes: {correct}/{total} = {accuracy:.4f}")
            print("\n" + "-" * 40 + "\n")
    except KeyboardInterrupt:
        print("\nBye!")

"""
Take an ImageNet class number (0-999, per classes.txt) as input, pick a
random training image from that class
(/mnt/data/Public_datasets/imagenet/imagenet_pytorch/train/), and show
CLIP's most probable label (and its probability) using five different
label sets: classes.txt, mod_classes.txt, openai_classes.txt,
mod_classes2.txt and openai_classes2.txt.
"""

import os
import random

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "openai/clip-vit-base-patch32"
CLASSES_PATH = "classes.txt"
IMAGENET_TRAIN_DIR = "/mnt/data/Public_datasets/imagenet/imagenet_pytorch/train"
LABEL_FILES = [
    "classes.txt",
    "mod_classes.txt",
    "openai_classes.txt",
    "mod_classes2.txt",
    "openai_classes2.txt",
]


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


def load_labels(path: str) -> list[tuple[int, str]]:
    labels = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            number, label = line.split(":", 1)
            labels.append((int(number), label.strip()))
    return labels


def load_wnids(train_dir: str = IMAGENET_TRAIN_DIR) -> list[str]:
    return sorted(os.listdir(train_dir))


def pick_random_image(class_number: int, wnids: list[str], train_dir: str = IMAGENET_TRAIN_DIR) -> str:
    wnid = wnids[class_number]
    class_dir = os.path.join(train_dir, wnid)
    filename = random.choice(os.listdir(class_dir))
    return os.path.join(class_dir, filename)


def load_clip(model_name: str = MODEL_NAME, device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor, device


def most_probable_label(image, labels, model, processor, device):
    prompts = [f"a photo of a {label}" for _, label in labels]
    inputs = processor(text=prompts, images=image, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1).squeeze(0)
    best_index = probs.argmax().item()
    number, label = labels[best_index]
    return number, label, probs[best_index].item()


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
            image_path = pick_random_image(class_number, wnids)
            image = Image.open(image_path).convert("RGB")

            print(f"\nTrue class: {class_number}: {classes[class_number]}")
            print(f"Random image: {image_path}\n")

            for label_file in LABEL_FILES:
                labels = load_labels(label_file)
                number, label, prob = most_probable_label(image, labels, model, processor, device)
                print(f"{label_file}: {number}: {label} (p={prob:.4f})")
            print("\n" + "-" * 40 + "\n")
    except KeyboardInterrupt:
        print("\nBye!")

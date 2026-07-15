"""
Take an ImageNet class number (per labels.txt) as input, gather every
validation image whose single reannotated label (from reanotace.jsonl)
matches that class, run them through CLIP zero-shot classification against
all labels.txt labels, and list the images where the true class is NOT
among CLIP's top 5 most probable labels.
"""

import json
import os

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "openai/clip-vit-base-patch32"
LABELS_PATH = "labels.txt"
JM_PATH = "jm.json"
REANNOTATION_PATH = "reanotace.jsonl"
IMAGENET_VAL_DIR = "/mnt/data/Public_datasets/imagenet/imagenet_pytorch/val"
TOP_K = 5


def load_labels(path: str = LABELS_PATH) -> list[tuple[int, str]]:
    labels = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            number, label = line.split(":", 1)
            labels.append((int(number), label.strip()))
    return labels


def load_merge_map(path: str = JM_PATH) -> dict[int, int]:
    """Map each dropped class number (the higher one in a jm.json pair) to
    the surviving number that appears in labels.txt (the lower one)."""
    merge_map = {}
    with open(path) as f:
        jm = json.load(f)
    for _, pair in jm["eq_classes"].items():
        keep, drop = min(pair), max(pair)
        merge_map[drop] = keep
    return merge_map


def load_clean_images(
    path: str, merge_map: dict[int, int], val_dir: str = IMAGENET_VAL_DIR
) -> dict[int, list[str]]:
    """Return {class_number: [image_path, ...]}, keeping only images that
    have exactly one reannotated label and that label isn't -1."""
    images_by_class: dict[int, list[str]] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            reannotated = entry["reannotated_labels"]
            if len(reannotated) != 1 or reannotated[0] == -1:
                continue
            label = merge_map.get(reannotated[0], reannotated[0])
            image_path = os.path.join(val_dir, entry["file_path"])
            images_by_class.setdefault(label, []).append(image_path)
    return images_by_class


def load_clip(model_name: str = MODEL_NAME, device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor, device


def predict_top_k_labels(image_paths, labels, model, processor, device, k: int = TOP_K):
    prompts = [f"a photo of a {label}" for _, label in labels]
    images = [Image.open(path).convert("RGB") for path in image_paths]
    inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)
    top_indices = probs.topk(k, dim=1).indices
    return [[labels[i][0] for i in indices.tolist()] for indices in top_indices]


def ask_class_number(images_by_class: dict[int, list[str]]) -> int:
    valid_numbers = sorted(images_by_class)
    while True:
        raw = input(
            f"Class number ({valid_numbers[0]}-{valid_numbers[-1]}, "
            "must have clean reannotated images): "
        ).strip()
        if not raw.isdigit() or int(raw) not in images_by_class:
            print("Please enter a class number that has clean reannotated images.")
            continue
        return int(raw)


if __name__ == "__main__":
    labels = load_labels()
    label_names = dict(labels)
    merge_map = load_merge_map()
    images_by_class = load_clean_images(REANNOTATION_PATH, merge_map)
    model, processor, device = load_clip()

    print("Press Ctrl+C to quit.\n")
    try:
        while True:
            class_number = ask_class_number(images_by_class)
            true_label = label_names[class_number]
            image_paths = images_by_class[class_number]

            top_k_numbers = predict_top_k_labels(image_paths, labels, model, processor, device)
            misses = [
                (path, numbers)
                for path, numbers in zip(image_paths, top_k_numbers)
                if class_number not in numbers
            ]

            print(f"\nClass {class_number}: {true_label} ({len(image_paths)} images)")
            if not misses:
                print(f"'{true_label}' was in the top {TOP_K} for all images.")
            else:
                print(f"{len(misses)} of {len(image_paths)} images did NOT have '{true_label}' in the top {TOP_K}:")
                for path, numbers in misses:
                    print(f"  {path}: {[label_names[n] for n in numbers]}")
            print("\n" + "-" * 40 + "\n")
    except KeyboardInterrupt:
        print("\nBye!")

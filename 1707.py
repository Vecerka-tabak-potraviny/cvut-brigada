"""
Take an ImageNet class number (per labels.txt) as input, gather every
validation image whose reannotated labels (from reanotace.jsonl) include
that class -- images with several reannotated labels count towards each
of those classes, and a -1 among several labels is simply ignored (only
a lone -1 label drops the image) -- and for each of zvlastni.txt and
my_labels.txt list the images where the chosen model's single most probable
label (top 1) for that label set isn't the true class, along with what that
top-1 label was. Before each run you pick whether CLIP or SigLIP2 does the
classifying.
"""

import json
import os

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, pipeline

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
SIGLIP_MODEL_NAME = "google/siglip2-base-patch16-224"
LABEL_SET_PATHS = ["zvlastni.txt", "my_labels.txt"]
JM_PATH = "jm.json"
REANNOTATION_PATH = "reanotace.jsonl"
IMAGENET_VAL_DIR = "/mnt/data/Public_datasets/imagenet/imagenet_pytorch/val"


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
    """Return {class_number: [image_path, ...]}, dropping only images whose
    sole reannotated label is -1. Any -1 alongside other labels is ignored,
    and an image with several valid labels is added to the list of every
    one of those classes."""
    images_by_class: dict[int, list[str]] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            reannotated = entry["reannotated_labels"]
            valid_labels = [label for label in reannotated if label != -1]
            if not valid_labels:
                continue
            image_path = os.path.join(val_dir, entry["file_path"])
            for raw_label in valid_labels:
                label = merge_map.get(raw_label, raw_label)
                images_by_class.setdefault(label, []).append(image_path)
    return images_by_class


def load_clip(model_name: str = CLIP_MODEL_NAME, device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = CLIPModel.from_pretrained(model_name).to(device).eval()
    except torch.AcceleratorError:
        print(f"Could not fit the model on {device} (likely out of memory); falling back to cpu.")
        device = "cpu"
        model = CLIPModel.from_pretrained(model_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor, device


def load_siglip(model_name: str = SIGLIP_MODEL_NAME, device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    try:
        image_classifier = pipeline(model=model_name, task="zero-shot-image-classification", device=device)
    except torch.AcceleratorError:
        print(f"Could not fit the model on {device} (likely out of memory); falling back to cpu.")
        device = "cpu"
        image_classifier = pipeline(model=model_name, task="zero-shot-image-classification", device=device)
    return image_classifier, device


def predict_top1_labels_clip(image_paths, labels, model, processor, device):
    prompts = [f"a photo of a {label}" for _, label in labels]
    images = [Image.open(path).convert("RGB") for path in image_paths]
    inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)
    top_index = probs.argmax(dim=1)
    return [labels[i][0] for i in top_index.tolist()]


def predict_top1_labels_siglip(image_paths, labels, image_classifier):
    label_texts = [label for _, label in labels]
    text_to_number = {label: number for number, label in labels}
    images = [Image.open(path).convert("RGB") for path in image_paths]
    results = image_classifier(images, candidate_labels=label_texts, hypothesis_template="a photo of a {}")
    return [text_to_number[image_results[0]["label"]] for image_results in results]


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


def ask_model_choice() -> str:
    while True:
        raw = input("Model to use (clip/siglip2): ").strip().lower()
        if raw in ("clip", "siglip2"):
            return raw
        print("Please enter 'clip' or 'siglip2'.")


if __name__ == "__main__":
    label_sets = {path: load_labels(path) for path in LABEL_SET_PATHS}
    merge_map = load_merge_map()
    images_by_class = load_clean_images(REANNOTATION_PATH, merge_map)
    backends: dict[str, tuple] = {}

    print("Press Ctrl+C to quit.\n")
    try:
        while True:
            model_choice = ask_model_choice()
            if model_choice not in backends:
                print(f"Loading {model_choice}...")
                backends[model_choice] = load_clip() if model_choice == "clip" else load_siglip()

            class_number = ask_class_number(images_by_class)
            image_paths = images_by_class[class_number]

            print(f"\nClass {class_number} ({len(image_paths)} images) -- model: {model_choice}")
            for path, labels in label_sets.items():
                label_names = dict(labels)
                true_label = label_names[class_number]
                if model_choice == "clip":
                    model, processor, device = backends["clip"]
                    top1_numbers = predict_top1_labels_clip(image_paths, labels, model, processor, device)
                else:
                    image_classifier, _ = backends["siglip2"]
                    top1_numbers = predict_top1_labels_siglip(image_paths, labels, image_classifier)
                misses = [
                    (image_path, number)
                    for image_path, number in zip(image_paths, top1_numbers)
                    if number != class_number
                ]
                print(f"  {path} ('{true_label}'): {len(misses)} of {len(image_paths)} images NOT top-1")
                for image_path, number in misses:
                    print(f"    {image_path}  ->  {label_names[number]}")
            print("\n" + "-" * 40 + "\n")
    except KeyboardInterrupt:
        print("\nBye!")

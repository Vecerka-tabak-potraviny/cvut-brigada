"""
Run every ImageNet validation image (all 1000 raw classes, from
/mnt/data/Public_datasets/imagenet/imagenet_pytorch/val/) through timm's
SigLIP2 (ViT-SO400M-16-SigLIP2-384, via open_clip), first against
zvlastni.txt and then against opai.txt, and report the overall Top-1
accuracy for each. zvlastni.txt only has 987 labels -- pairs of raw
classes that jm.json merged into the same label are pooled together
(their images combined into one group) before scoring; opai.txt has all
1000 raw classes 1:1, so no merging is needed there.

Classification follows the zero-shot-classifier technique from a.ipynb
(OpenAI's CLIP ImageNet notebook): for each class, embed it under all 7
prompt templates, L2-normalize each embedding, average them, and
L2-normalize the average to get one "classifier weight" vector per class.
An image's predicted class is whichever class weight vector its (also
normalized) image embedding has the highest cosine similarity with --
instead of the single "a photo of a {label}" prompt used in the other
scripts here.
"""

import json
import os

import torch
from open_clip import create_model_from_pretrained, get_tokenizer
from PIL import Image
from tqdm import tqdm

MODEL_NAME = "hf-hub:timm/ViT-SO400M-16-SigLIP2-384"
ZVLASTNI_LABELS_PATH = "zvlastni.txt"
OPAI_LABELS_PATH = "opai.txt"
JM_PATH = "jm.json"
IMAGENET_VAL_DIR = "/mnt/data/Public_datasets/imagenet/imagenet_pytorch/val"

# The 7-template subset from a.ipynb (OpenAI's CLIP ImageNet notebook),
# found via sequential forward selection over the full 80 templates.
TEMPLATES = [
    "itap of a {}.",
    "a bad photo of the {}.",
    "a origami {}.",
    "a photo of the large {}.",
    "a {} in a video game.",
    "art of the {}.",
    "a photo of the small {}.",
]


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
    """Map each dropped raw class number (the higher one in a jm.json pair)
    to the surviving number that appears in zvlastni.txt (the lower one)."""
    merge_map = {}
    with open(path) as f:
        jm = json.load(f)
    for _, pair in jm["eq_classes"].items():
        keep, drop = min(pair), max(pair)
        merge_map[drop] = keep
    return merge_map


def load_wnids(val_dir: str = IMAGENET_VAL_DIR) -> list[str]:
    return sorted(os.listdir(val_dir))


def images_for_class(class_number: int, wnids: list[str], val_dir: str = IMAGENET_VAL_DIR) -> list[str]:
    wnid = wnids[class_number]
    class_dir = os.path.join(val_dir, wnid)
    return [os.path.join(class_dir, name) for name in sorted(os.listdir(class_dir))]


def images_by_merged_class(
    wnids: list[str], merge_map: dict[int, int], val_dir: str = IMAGENET_VAL_DIR
) -> dict[int, list[str]]:
    """Return {merged_class_number: [image_path, ...]}, pooling together the
    images of every raw class that jm.json merged into the same number."""
    grouped: dict[int, list[str]] = {}
    for raw_class_number in range(len(wnids)):
        true_number = merge_map.get(raw_class_number, raw_class_number)
        grouped.setdefault(true_number, []).extend(images_for_class(raw_class_number, wnids, val_dir))
    return grouped


def load_siglip(model_name: str = MODEL_NAME, device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model, preprocess = create_model_from_pretrained(model_name, device=device)
    except torch.AcceleratorError:
        print(f"Could not fit the model on {device} (likely out of memory); falling back to cpu.")
        device = "cpu"
        model, preprocess = create_model_from_pretrained(model_name, device=device)
    model.eval()
    tokenizer = get_tokenizer(model_name)
    return model, preprocess, tokenizer, device


def build_zeroshot_weights(labels, templates, model, tokenizer, device):
    """One L2-normalized classifier weight vector per class, averaged over
    every prompt template (the a.ipynb zeroshot_classifier technique).
    encode_text(..., normalize=True) already L2-normalizes each embedding."""
    weights = []
    with torch.no_grad():
        for _, label in tqdm(labels, desc="Building zero-shot classifier"):
            texts = [template.format(label) for template in templates]
            tokens = tokenizer(texts, context_length=model.context_length).to(device)
            embeddings = model.encode_text(tokens, normalize=True)
            embedding = embeddings.mean(dim=0)
            embedding = embedding / embedding.norm()
            weights.append(embedding)
    return torch.stack(weights, dim=1)


def predict_top1_numbers(image_paths, labels, zeroshot_weights, model, preprocess, device):
    images = [preprocess(Image.open(path).convert("RGB")) for path in image_paths]
    pixel_values = torch.stack(images).to(device)
    with torch.no_grad():
        image_features = model.encode_image(pixel_values, normalize=True)
    logits = image_features @ zeroshot_weights
    top_index = logits.argmax(dim=1)
    return [labels[i][0] for i in top_index.tolist()]


def run_accuracy(name, labels, images_by_class, model, preprocess, tokenizer, device):
    label_names = dict(labels)
    zeroshot_weights = build_zeroshot_weights(labels, TEMPLATES, model, tokenizer, device)

    correct = 0
    total = 0
    progress = tqdm(sorted(images_by_class), desc=f"[{name}] Top-1 accuracy: n/a")
    for true_number in progress:
        image_paths = images_by_class[true_number]
        predicted_numbers = predict_top1_numbers(image_paths, labels, zeroshot_weights, model, preprocess, device)

        class_correct = sum(1 for number in predicted_numbers if number == true_number)
        correct += class_correct
        total += len(image_paths)
        progress.set_description(f"[{name}] Top-1 accuracy: {correct / total:.4f}")
        progress.write(
            f"[{name}] Class {true_number} ({label_names[true_number]}): "
            f"{class_correct}/{len(image_paths)} correct"
        )

    print(f"\n[{name}] Overall Top-1 accuracy over {total} images: {correct}/{total} = {correct / total:.4f}\n")


if __name__ == "__main__":
    wnids = load_wnids()
    model, preprocess, tokenizer, device = load_siglip()

    zvlastni_labels = load_labels(ZVLASTNI_LABELS_PATH)
    merge_map = load_merge_map()
    zvlastni_images = images_by_merged_class(wnids, merge_map)
    run_accuracy("zvlastni.txt", zvlastni_labels, zvlastni_images, model, preprocess, tokenizer, device)

    opai_labels = load_labels(OPAI_LABELS_PATH)
    opai_images = {class_number: images_for_class(class_number, wnids) for class_number in range(len(wnids))}
    run_accuracy("opai.txt", opai_labels, opai_images, model, preprocess, tokenizer, device)

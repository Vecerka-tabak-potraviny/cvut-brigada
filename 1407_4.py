"""
Take an image path as input and show CLIP's three most probable labels
(and their probabilities) from mod_classes2.txt.
"""

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "openai/clip-vit-base-patch32"
LABELS_PATH = "motyle.txt"


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


def load_clip(model_name: str = MODEL_NAME, device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor, device


def top_k_labels(image, labels, model, processor, device, k: int = 3):
    prompts = [f"a photo of a {label}" for _, label in labels]
    inputs = processor(text=prompts, images=image, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1).squeeze(0)
    top_probs, top_indices = probs.topk(k)
    return [(labels[i][0], labels[i][1], p.item()) for i, p in zip(top_indices.tolist(), top_probs)]


def ask_image_path() -> str:
    while True:
        raw = input("Image path: ").strip()
        try:
            with Image.open(raw):
                return raw
        except (FileNotFoundError, OSError) as e:
            print(f"Could not open image: {e}")


if __name__ == "__main__":
    labels = load_labels()
    model, processor, device = load_clip()

    print("Press Ctrl+C to quit.\n")
    try:
        while True:
            image_path = ask_image_path()
            image = Image.open(image_path).convert("RGB")

            results = top_k_labels(image, labels, model, processor, device, k=3)

            print(f"\nTop 3 labels for {image_path}:")
            for number, label, prob in results:
                print(f"  {number}: {label}: {prob:.4f}")
            print("\n" + "-" * 40 + "\n")
    except KeyboardInterrupt:
        print("\nBye!")

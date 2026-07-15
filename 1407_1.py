"""
Take an ImageNet class number (0-999, per classes.txt) as input and list
the validation images belonging to that class in the local ImageNet
dataset at /mnt/data/Public_datasets/imagenet/imagenet_pytorch/val/.
"""

import os

CLASSES_PATH = "classes.txt"
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

    print("Press Ctrl+C to quit.\n")
    try:
        while True:
            class_number = ask_class_number(classes)
            label = classes[class_number]
            paths = images_for_class(class_number, wnids)

            print(f"\nClass {class_number}: {label} ({len(paths)} images)")
            for path in paths:
                print(f"  {path}")
            print("\n" + "-" * 40 + "\n")
    except KeyboardInterrupt:
        print("\nBye!")

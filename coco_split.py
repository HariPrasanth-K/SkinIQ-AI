import json
import os
import shutil
import random
from pathlib import Path

def combine_and_split_coco(
    dataset_dirs: list[dict],
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42
):
    """
    Combine multiple COCO datasets and split into train/valid/test.

    Output layout matches RF-DETR COCO format:
        output_dir/
        ├── train/
        │   ├── _annotations.coco.json
        │   ├── image1.jpg
        │   └── ...
        ├── valid/
        │   ├── _annotations.coco.json
        │   └── ...
        └── test/
            ├── _annotations.coco.json
            └── ...
    
    Args:
        dataset_dirs: List of dicts with keys:
                      - 'images_dir': path to images folder
                      - 'annotations': path to annotations JSON file
        output_dir: Path to save the combined and split dataset
        train_ratio: Fraction for training set
        val_ratio: Fraction for validation set
        test_ratio: Fraction for test set
        seed: Random seed for reproducibility
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    random.seed(seed)

    # ─── Merge all datasets ────────────────────────────────────────────────────
    merged = {
        "info": {},
        "licenses": [],
        "categories": [],
        "images": [],
        "annotations": []
    }

    category_name_to_id = {}   # unified category map
    next_category_id   = 1
    next_image_id      = 1
    next_annotation_id = 1

    # Maps old image ids -> new image id (per dataset)
    all_image_records = []   # (new_image_record, src_image_path)

    for ds in dataset_dirs:
        ann_path    = ds["annotations"]
        images_dir  = ds["images_dir"]

        print(f"Loading annotations from: {ann_path}")
        with open(ann_path) as f:
            coco = json.load(f)

        # Merge info / licenses from first dataset
        if not merged["info"]:
            merged["info"]     = coco.get("info", {})
            merged["licenses"] = coco.get("licenses", [])

        # ── Remap categories ──────────────────────────────────────────────────
        old_cat_id_to_new = {}
        for cat in coco.get("categories", []):
            name = cat["name"]
            if name not in category_name_to_id:
                category_name_to_id[name] = next_category_id
                merged["categories"].append({
                    "id":           next_category_id,
                    "name":         name,
                    "supercategory": cat.get("supercategory", "")
                })
                next_category_id += 1
            old_cat_id_to_new[cat["id"]] = category_name_to_id[name]

        # ── Remap images ──────────────────────────────────────────────────────
        old_img_id_to_new = {}
        for img in coco.get("images", []):
            new_id = next_image_id
            old_img_id_to_new[img["id"]] = new_id
            next_image_id += 1

            new_img = {**img, "id": new_id}
            all_image_records.append((new_img, os.path.join(images_dir, img["file_name"])))

        # ── Remap annotations ─────────────────────────────────────────────────
        for ann in coco.get("annotations", []):
            new_ann = {
                **ann,
                "id":           next_annotation_id,
                "image_id":     old_img_id_to_new[ann["image_id"]],
                "category_id":  old_cat_id_to_new[ann["category_id"]]
            }
            merged["annotations"].append(new_ann)
            next_annotation_id += 1

    print(f"\nTotal images     : {len(all_image_records)}")
    print(f"Total annotations: {len(merged['annotations'])}")
    print(f"Total categories : {len(merged['categories'])}")

    # ─── Split ─────────────────────────────────────────────────────────────────
    random.shuffle(all_image_records)
    n       = len(all_image_records)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    # RF-DETR expects train / valid / test (not "val")
    splits = {
        "train":  all_image_records[:n_train],
        "valid":  all_image_records[n_train:n_train + n_val],
        "test":   all_image_records[n_train + n_val:]
    }

    # Build annotation lookup by image_id
    ann_by_image = {}
    for ann in merged["annotations"]:
        ann_by_image.setdefault(ann["image_id"], []).append(ann)

    # ─── Save each split (RF-DETR layout: images + _annotations.coco.json in same folder) ──
    for split_name, records in splits.items():
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        split_coco = {
            "info":        merged["info"],
            "licenses":    merged["licenses"],
            "categories":  merged["categories"],
            "images":      [],
            "annotations": []
        }

        for img_record, src_path in records:
            split_coco["images"].append(img_record)
            split_coco["annotations"].extend(ann_by_image.get(img_record["id"], []))

            # Copy image directly into split folder (no images/ subdir)
            dst_path = os.path.join(split_dir, img_record["file_name"])
            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)
            else:
                print(f"  [WARNING] Image not found: {src_path}")

        ann_file = os.path.join(split_dir, "_annotations.coco.json")
        with open(ann_file, "w") as f:
            json.dump(split_coco, f, indent=2)

        print(f"\n{split_name.upper():5s} → {len(split_coco['images'])} images, "
              f"{len(split_coco['annotations'])} annotations")
        print(f"       Saved to: {split_dir} (_annotations.coco.json + images)")

    print("\nDone!")


# ──────────────────────────────────────────────────────────────────────────────
# USAGE EXAMPLE
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    datasets = [
        {
            "images_dir":  r"C:\Users\HARI PRASANTH K\Downloads\odfml (2)\odfml\coco_input_folders\pigment-1\images",
            "annotations": r"C:\Users\HARI PRASANTH K\Downloads\odfml (2)\odfml\coco_input_folders\pigment-1\result.json"
        },
        {
            "images_dir":  r"C:\Users\HARI PRASANTH K\Downloads\odfml (2)\odfml\coco_input_folders\pigment-2\images",
            "annotations": r"C:\Users\HARI PRASANTH K\Downloads\odfml (2)\odfml\coco_input_folders\pigment-2\result.json"
        },
    ]

    combine_and_split_coco(
        dataset_dirs = datasets,
        output_dir   = r"C:\Users\HARI PRASANTH K\Downloads\odfml (2)\odfml\coco_data_split",
        train_ratio  = 0.70,
        val_ratio    = 0.20,
        test_ratio   = 0.10,
        seed         = 42
    )

import json
import random
import os
import shutil
import argparse

def create_coco_overfit_subset(coco_json_path, images_dir, output_json_path, output_images_dir, num_images=10):
    # Load COCO JSON
    with open(coco_json_path, 'r') as f:
        coco = json.load(f)

    # Randomly select N images
    selected_images = random.sample(coco["images"], num_images)
    selected_ids = {img["id"] for img in selected_images}

    # Filter annotations
    selected_annotations = [ann for ann in coco["annotations"] if ann["image_id"] in selected_ids]

    # Copy images to new folder
    os.makedirs(output_images_dir, exist_ok=True)
    for img in selected_images:
        src = os.path.join(images_dir, img["file_name"])
        dst = os.path.join(output_images_dir, img["file_name"])
        if os.path.exists(src):
            shutil.copy(src, dst)
        else:
            print(f"Warning: {src} not found.")

    # Create and save overfit JSON
    new_coco = {
        "images": selected_images,
        "annotations": selected_annotations,
        "categories": coco.get("categories", []),
    }

    with open(output_json_path, 'w') as f:
        json.dump(new_coco, f, indent=2)

    print(f"✅ Saved {len(selected_images)} images and {len(selected_annotations)} annotations to {output_json_path}")
    print(f"🖼️  Copied images to: {output_images_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a COCO overfit subset with 10 random images.")
    parser.add_argument("--coco_json", required=True, help="Path to original COCO JSON file.")
    parser.add_argument("--images_dir", required=True, help="Directory containing all images.")
    parser.add_argument("--output_json", default="coco_overfit.json", help="Path to save new COCO JSON.")
    parser.add_argument("--output_images_dir", default="overfit_images", help="Directory to save selected images.")
    parser.add_argument("--num_images", type=int, default=10, help="Number of images to select.")

    args = parser.parse_args()

    create_coco_overfit_subset(
        coco_json_path=args.coco_json,
        images_dir=args.images_dir,
        output_json_path=args.output_json,
        output_images_dir=args.output_images_dir,
        num_images=args.num_images
    )

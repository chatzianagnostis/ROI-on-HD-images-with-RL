import matplotlib # Βεβαιώσου ότι έχεις κάνει import
from ROIDataset import ROIDataset # Κάνε import τις κλάσεις σου
from ROIDetectionEnv import ROIDetectionEnv
# from agent import ROIAgent # Προαιρετικά, αν θέλεις να δεις την οπτική ενός εκπαιδευμένου agent
import os

os.environ['LOKY_MAX_CPU_COUNT'] = '4'

def main(): # Άλλαξε το όνομα της συνάρτησης
    print("Visualizing Reward Landscape for multiple images...")

    dataset_path = "G:\\rl\\overfit\\images"
    coco_json_path = "G:\\rl\\overfit\\overfit.json"

    dataset = ROIDataset(
        dataset_path=dataset_path,
        coco_json_path=coco_json_path,
        image_size=(640, 640),
        annotations_format="coco",
        shuffle=False # ΣΗΜΑΝΤΙΚΟ: Βάλε shuffle=False για να είναι προβλέψιμη η σειρά
    )

    if len(dataset) == 0:
        print("Dataset is empty.")
        return

    env = ROIDetectionEnv(
        dataset=dataset,
        crop_size=(640, 640),
        time_limit=120
    )

    output_dir = "landscape_visualizations_all" # Νέος φάκελος για τις πολλές εικόνες
    os.makedirs(output_dir, exist_ok=True)

    # Βρόχος για να περάσεις από κάποιες ή όλες τις εικόνες
    num_images_to_process = len(dataset) # Επεξεργάσου όλες τις εικόνες
    # ή num_images_to_process = 5 # Για τις πρώτες 5

    for i in range(num_images_to_process):
        print(f"\nProcessing image {i+1}/{num_images_to_process}...")
        try:
            observation = env.reset() # Φορτώνει την επόμενη εικόνα
                                    # (ή την πρώτη αν είναι η πρώτη επανάληψη μετά το shuffle=False)
            current_image_id = env.current_sample['image_id']
            print(f"Current image loaded: {current_image_id}")
            print(f"Calculated bbox_size for visualization: {env.bbox_size}")

            image_id_for_filename = os.path.splitext(current_image_id)[0]
            output_path = os.path.join(output_dir, f"reward_landscape_{image_id_for_filename}.jpg")

            print(f"Generating reward landscape and saving to: {output_path}")
            env.visualize_reward_landscape(output_path=output_path)
            print(f"Visualization for {current_image_id} complete.")

        except StopIteration: # Αν το dataset τελειώσει νωρίτερα από το num_images_to_process
            print("Reached end of dataset.")
            break
        except Exception as e:
            print(f"An error occurred processing an image: {e}")
            # Προαιρετικά, μπορείς να συνεχίσεις με την επόμενη εικόνα
            # continue

if __name__ == "__main__":
    main()
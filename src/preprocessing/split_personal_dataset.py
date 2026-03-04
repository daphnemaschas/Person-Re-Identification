import os
import shutil
import yaml
import random
from collections import defaultdict

class ReIDDatasetSplitter:
    """Splits processed person crops into query and bounding_box_test folders.

    Attributes:
        config (dict): Dictionary containing dataset paths from the config file.
        query_dir (str): Path to the output query directory.
        gallery_dir (str): Path to the output bounding_box_test directory.
    """

    def __init__(self, config_path: str):
        """Initializes the splitter with parameters from a YAML config file.

        Args:
            config_path (str): Path to the configuration file.
        """
        with open(config_path, "r") as f:
            full_config = yaml.safe_load(f)["reid_project"]
        
        self.config = full_config["personal_data"]
        
        base_dir = os.path.dirname(self.config["detected_crops_dir"])
        self.query_dir = os.path.join(base_dir, "query")
        self.gallery_dir = os.path.join(base_dir, "bounding_box_test")
        
        os.makedirs(self.query_dir, exist_ok=True)
        os.makedirs(self.gallery_dir, exist_ok=True)

    def run_split(self):
        """Processes each identity folder and splits images."""
        src_root = self.config["detected_crops_dir"]
        
        for person_id in os.listdir(src_root):
            person_path = os.path.join(src_root, person_id)
            if not os.path.isdir(person_path):
                continue
            
            images = [f for f in os.listdir(person_path) if f.endswith(".jpg")]
            if not images:
                continue

            cam_groups = defaultdict(list)
            for img in images:
                cam_id = img.split('_')[1]
                cam_groups[cam_id].append(img)

            self._distribute_images(person_path, cam_groups)

    def _distribute_images(self, person_path: str, cam_groups: dict):
        """Distributes files physically based on camera grouping.

        Args:
            person_path (str): Source directory of the person's images.
            cam_groups (dict): Dictionary mapping camera IDs to image names.
        """
        for cam_id, imgs in cam_groups.items():
            query_img = random.choice(imgs)
            shutil.copy(os.path.join(person_path, query_img), 
                        os.path.join(self.query_dir, query_img))
            
            for img in imgs:
                if img == query_img:
                    continue
                shutil.copy(os.path.join(person_path, img), 
                            os.path.join(self.gallery_dir, img))

if __name__ == "__main__":
    splitter = ReIDDatasetSplitter("config.yaml")
    splitter.run_split()
    print("Split completed successfully. Folders 'query' and 'bounding_box_test' are ready.")
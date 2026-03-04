import os
import cv2
import yaml
from ultralytics import YOLO
from tqdm import tqdm

class ReIDDatasetExtractor:
    """Class to extract person crops from raw images using YOLO for Re-ID datasets.
    
    Attributes:
        config: Dictionary containing preprocessing parameters.
        model: YOLO model instance.
        target_ratio: Aspect ratio (H/W) for the output crops.
    """

    def __init__(self, config_path: str):
        """Initializes the extractor with parameters from a YAML config file.

        Args:
            config_path: Path to the configuration file.
        """
        with open(config_path, "r") as f:
            full_config = yaml.safe_load(f)["reid_project"]
        
        self.config = full_config["personal_data"]

        img_size = full_config["market1501"]["img_size"]
        self.target_ratio = img_size[0] / img_size[1]
        
        self.model = YOLO(self.config["yolo_model"])
        self.person_class_id = 0

    def _get_agnostic_crop(self, image, box):
        """Calculates coordinates to crop image while maintaining target aspect ratio.

        Args:
            image: Input image as numpy array.
            box: YOLO bounding box coordinates [x1, y1, x2, y2].

        Returns:
            Cropped image as numpy array.
        """
        img_h, img_w = image.shape[:2]
        x1, y1, x2, y2 = map(int, box)
        w, h = x2 - x1, y2 - y1
        
        if h / w > self.target_ratio:
            new_w, new_h = h / self.target_ratio, h
        else:
            new_h, new_w = w * self.target_ratio, w
            
        center_x, center_y = x1 + w / 2, y1 + h / 2
        nx1 = int(max(0, center_x - new_w / 2))
        ny1 = int(max(0, center_y - new_h / 2))
        nx2 = int(min(img_w, center_x + new_w / 2))
        ny2 = int(min(img_h, center_y + new_h / 2))
        
        return image[ny1:ny2, nx1:nx2]

    def process_all(self):
        """Iterates through cameras and persons to extract crops.
        
        Expected structure: raw_photos_dir/camera_XX/personne_XX/
        """
        raw_dir = self.config["raw_photos_dir"]
        if not os.path.exists(raw_dir):
            print(f"Directory {raw_dir} not found.")
            return

        for cam_folder in os.listdir(raw_dir):
            cam_path = os.path.join(raw_dir, cam_folder)
            if not os.path.isdir(cam_path):
                continue
            
            for pers_folder in os.listdir(cam_path):
                pers_path = os.path.join(cam_path, pers_folder)
                if not os.path.isdir(pers_path):
                    continue
                
                self._process_person_folder(pers_path, pers_folder, cam_folder)

    def _process_person_folder(self, folder_path: str, pers_name: str, cam_name: str):
        """Extracts person crops from a specific directory.

        Args:
            folder_path: Path to the images.
            pers_name: Name of the person folder.
            cam_name: Name of the camera folder.
        """
        try:
            p_id = f"{int(''.join(filter(str.isdigit, pers_name))):04d}"
            c_id = f"c{int(''.join(filter(str.isdigit, cam_name)))}"
        except ValueError:
            p_id, c_id = pers_name, cam_name

        dest_dir = os.path.join(self.config["detected_crops_dir"], p_id)
        os.makedirs(dest_dir, exist_ok=True)

        images = [f for f in os.listdir(folder_path) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for i, filename in enumerate(tqdm(images, desc=f"ID:{p_id} ({c_id})")):
            img = cv2.imread(os.path.join(folder_path, filename))
            if img is None:
                continue
            
            results = self.model(img, verbose=False, conf=self.config["detection_threshold"])
            
            for result in results:
                for box in result.boxes:
                    if int(box.cls) == self.person_class_id:
                        crop = self._get_agnostic_crop(img, box.xyxy[0].cpu().numpy())
                        out_name = f"{p_id}_{c_id}s1_{i:06d}_00.jpg"
                        cv2.imwrite(os.path.join(dest_dir, out_name), crop)
                        break 

if __name__ == "__main__":
    extractor = ReIDDatasetExtractor("config.yaml")
    extractor.process_all()
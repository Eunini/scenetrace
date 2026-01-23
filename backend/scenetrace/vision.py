"""Real frozen CLIP retrieval and DETR boxes, loaded lazily under a shared lock."""

import threading
from pathlib import Path

import numpy as np
from PIL import Image

CLIP_MODEL = "openai/clip-vit-base-patch32"
CLIP_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
DETR_MODEL = "facebook/detr-resnet-50"
DETR_REVISION = "1d5f47bd3bdd2c4bbfa585418ffe6da5028b4c0b"


def normalized(value):
    array = np.asarray(value, dtype=np.float32)
    return array / np.maximum(np.linalg.norm(array, axis=-1, keepdims=True), 1e-8)


class Vision:
    def __init__(self):
        self.lock = threading.RLock()
        self.model = self.processor = self.detector = self.detector_processor = None

    def load(self):
        if self.model is not None:
            return
        import torch
        from transformers import CLIPModel, CLIPProcessor

        torch.set_num_threads(2)
        self.processor = CLIPProcessor.from_pretrained(CLIP_MODEL, revision=CLIP_REVISION)
        self.model = CLIPModel.from_pretrained(
            CLIP_MODEL, revision=CLIP_REVISION, use_safetensors=True
        ).eval()

    def images(self, paths):
        import torch

        with self.lock, torch.inference_mode():
            self.load()
            outputs = []
            for start in range(0, len(paths), 4):
                images = [Image.open(path).convert("RGB") for path in paths[start:start + 4]]
                inputs = self.processor(images=images, return_tensors="pt")
                features = self.model.get_image_features(**inputs)
                if hasattr(features, "pooler_output"):
                    features = features.pooler_output
                outputs.append(normalized(features.cpu().numpy()))
                for image in images:
                    image.close()
            return np.concatenate(outputs)

    def text(self, query):
        import torch

        with self.lock, torch.inference_mode():
            self.load()
            inputs = self.processor(text=[query], return_tensors="pt", padding=True, truncation=True)
            features = self.model.get_text_features(**inputs)
            if hasattr(features, "pooler_output"):
                features = features.pooler_output
            return normalized(features.cpu().numpy())[0]

    def detect(self, image_path: Path):
        import torch
        from transformers import DetrForObjectDetection, DetrImageProcessor

        with self.lock, torch.inference_mode(), Image.open(image_path) as image:
            image = image.convert("RGB")
            if self.detector is None:
                self.detector_processor = DetrImageProcessor.from_pretrained(DETR_MODEL, revision=DETR_REVISION)
                self.detector = DetrForObjectDetection.from_pretrained(
                    DETR_MODEL, revision=DETR_REVISION, use_safetensors=True,
                    backbone_kwargs={"use_pretrained_backbone": False},
                ).eval()
            inputs = self.detector_processor(images=image, return_tensors="pt")
            outputs = self.detector(**inputs)
            result = self.detector_processor.post_process_object_detection(
                outputs, target_sizes=torch.tensor([[image.height, image.width]]), threshold=0.7
            )[0]
            return {"width": image.width, "height": image.height, "objects": [
                {"label": self.detector.config.id2label[int(label)], "score": round(float(score), 4),
                 "box": [round(float(x), 2) for x in box]}
                for score, label, box in zip(result["scores"], result["labels"], result["boxes"], strict=True)
            ]}


def sharpness(path: Path):
    import cv2

    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2)


def object_mask(image_path: Path, destination: Path, box):
    """Classical GrabCut initialized from a DETR/user box; not a learned segmentation model."""
    import cv2

    image = cv2.imread(str(image_path))
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box
    x, y = max(1, int(x1)), max(1, int(y1))
    width = min(w - x - 1, max(2, int(x2 - x1)))
    height = min(h - y - 1, max(2, int(y2 - y1)))
    if width <= 1 or height <= 1 or x >= w - 2 or y >= h - 2:
        raise ValueError("Choose a bounding box within the frame.")
    mask = np.zeros((h, w), np.uint8)
    cv2.grabCut(image, mask, (x, y, width, height), np.zeros((1, 65)), np.zeros((1, 65)), 3, cv2.GC_INIT_WITH_RECT)
    alpha = np.where((mask == 1) | (mask == 3), 255, 0).astype(np.uint8)
    rgba = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = alpha
    cv2.imwrite(str(destination), rgba)
    return {"foreground_pixels": int(np.count_nonzero(alpha)), "total_pixels": w * h,
            "method": "GrabCut, initialized from a bounding box"}

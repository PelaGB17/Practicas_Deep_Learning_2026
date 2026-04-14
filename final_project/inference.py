import io
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image

from final_project.data import build_eval_transform
from final_project.modeling import build_resnet50, resolve_device
from final_project.settings import IMAGENET_MEAN, IMAGENET_STD


class ScenePredictor:
    def __init__(
        self,
        checkpoint_path: Path,
        device: str = "auto",
        class_names: Optional[List[str]] = None,
    ) -> None:
        self.device = resolve_device(device)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        if "model_state_dict" not in checkpoint:
            raise ValueError(
                "Checkpoint format not supported. Expected key 'model_state_dict'."
            )

        self.class_names = class_names or checkpoint.get("class_names")
        if not self.class_names:
            raise ValueError(
                "Class names are missing. Provide class_names or train a checkpoint with metadata."
            )

        image_size = int(checkpoint.get("image_size", 224))
        mean = tuple(checkpoint.get("mean", IMAGENET_MEAN))
        std = tuple(checkpoint.get("std", IMAGENET_STD))

        self.transform = build_eval_transform(image_size=image_size, mean=mean, std=std)
        self.model = build_resnet50(
            num_classes=len(self.class_names),
            unfreeze_blocks=0,
            pretrained=False,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict_pil(self, image: Image.Image, top_k: int = 3) -> Dict[str, object]:
        image = image.convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        logits = self.model(tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu()

        k = min(top_k, len(self.class_names))
        top_values, top_indices = torch.topk(probabilities, k)

        top_predictions = []
        for value, index in zip(top_values.tolist(), top_indices.tolist()):
            top_predictions.append(
                {
                    "label": self.class_names[index],
                    "probability": float(value),
                }
            )

        return {
            "label": top_predictions[0]["label"],
            "confidence": top_predictions[0]["probability"],
            "top_k": top_predictions,
        }

    def predict_bytes(self, image_bytes: bytes, top_k: int = 3) -> Dict[str, object]:
        image = Image.open(io.BytesIO(image_bytes))
        return self.predict_pil(image=image, top_k=top_k)

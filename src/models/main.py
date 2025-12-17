import segmentation_models_pytorch as smp

from abc import abstractmethod
import torch

from ..datasets import CityscapesUNetDataset, CityscapesSegFormerDataset
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation


def from_type(type):
    if type == "unet":
        return (
            smp.Unet(
                encoder_name="resnet50",
                encoder_weights="imagenet",
                classes=20,
                activation=None,
            ),
            CityscapesUNetDataset,
        )
    elif type == "segformer":

        return (
            SegformerForSemanticSegmentation.from_pretrained(
                "nvidia/segformer-b2-finetuned-ade-512-512",
                num_labels=20,
                ignore_mismatched_sizes=True,
            ),
            CityscapesSegFormerDataset,
        )

    else:
        raise ValueError(f"Unknown model type: {type}")

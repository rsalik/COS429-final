import segmentation_models_pytorch as smp

from abc import abstractmethod
import torch

from ..datasets import CityscapesDataset


def from_type(type, tvt=0):
    if type == "unet":
        return smp.Unet(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            classes=20,
            activation=None,
        ), CityscapesDataset(tvt)
    elif type == "segformer":

        raise ValueError("Segformer not implemented")

    else:
        raise ValueError(f"Unknown model type: {type}")

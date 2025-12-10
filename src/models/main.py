import segmentation_models_pytorch as smp

from abc import abstractmethod
import torch

from ..datasets import CityscapesDataset

def from_type(type, tvt=0):
    if type == "unet":
        return smp.Unet(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            classes=19,
            activation=None,
        ), CityscapesDataset(tvt)
    elif type == "segformer":
       
        processor = SegformerFeatureExtractor.from_pretrained(
            "nvidia/segformer-b2-finetuned-ade-512-512"
        )

        model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b2-finetuned-ade-512-512",
            num_labels=19,
            ignore_mismatched_sizes=True,
        )
    
        dataset = SegFormerCityscapesDataset(tvt, processor)
        
        return model, dataset
        
    else:
        raise ValueError(f"Unknown model type: {type}")

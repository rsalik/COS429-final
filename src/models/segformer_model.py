from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor

def load_segformer():
    """
    Loads SegFormer-B2 with 19 output classes.
    Returns: (model, processor)
    """
    processor = SegformerFeatureExtractor.from_pretrained(
        "nvidia/segformer-b2-finetuned-ade-512-512"
    )

    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b2-finetuned-ade-512-512",
        num_labels=19,
        ignore_mismatched_sizes=True,
    )

    return model, processor

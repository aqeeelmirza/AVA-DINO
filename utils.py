import torch
from CLIP.tokenizer import tokenize


def encode_text_with_prompt_ensemble(model, obj, device, cnn, adapter):
    """
    Encode an object class into normal/anomaly text embeddings via prompt ensembling.

    Generates a large set of prompt sentences for both normal and anomalous states,
    encodes them with CLIP, and returns the mean-normalized embedding per state.

    Args:
        model:   CLIP model with encode_text()
        obj:     object/class name string (e.g. "bottle", "capsule")
        device:  torch.device
        cnn:     passed through to model.encode_text (legacy arg)
        adapter: passed through to model.encode_text (legacy arg)

    Returns:
        text_features: [768, 2] tensor — column 0 = normal, column 1 = anomaly
    """
    prompt_normal   = ['{}', 'flawless {}', 'perfect {}', 'unblemished {}',
                       '{} without flaw', '{} without defect', '{} without damage']
    prompt_abnormal = ['damaged {}', 'broken {}', '{} with flaw',
                       '{} with defect', '{} with damage']
    prompt_templates = [
        'a bad photo of a {}.', 'a low resolution photo of the {}.', 'a bad photo of the {}.',
        'a cropped photo of the {}.', 'a bright photo of a {}.', 'a dark photo of the {}.',
        'a photo of my {}.', 'a photo of the cool {}.', 'a close-up photo of a {}.',
        'a black and white photo of the {}.', 'a bright photo of the {}.', 'a cropped photo of a {}.',
        'a jpeg corrupted photo of a {}.', 'a blurry photo of the {}.', 'a photo of the {}.',
        'a good photo of the {}.', 'a photo of one {}.', 'a close-up photo of the {}.',
        'a photo of a {}.', 'a low resolution photo of a {}.', 'a photo of a large {}.',
        'a blurry photo of a {}.', 'a jpeg corrupted photo of the {}.', 'a good photo of a {}.',
        'a photo of the small {}.', 'a photo of the large {}.', 'a black and white photo of a {}.',
        'a dark photo of a {}.', 'a photo of a cool {}.', 'a photo of a small {}.',
        'there is a {} in the scene.', 'there is the {} in the scene.',
        'this is a {} in the scene.', 'this is the {} in the scene.', 'this is one {} in the scene.',
    ]

    text_features = []
    for prompts in [prompt_normal, prompt_abnormal]:
        sentences = [
            template.format(state.format(obj))
            for state in prompts
            for template in prompt_templates
        ]
        tokens = tokenize(sentences).to(device)
        with torch.no_grad():
            embeddings = model.encode_text(text=tokens, cnn=cnn, device=device, adapter=adapter)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        mean_emb = embeddings.mean(dim=0)
        text_features.append(mean_emb / mean_emb.norm())

    return torch.stack(text_features, dim=1).to(device)

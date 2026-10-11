import torch
import torch.nn.functional as F


def semantic_aggregation(logits, masks):
    return torch.einsum('bqc,bqhw->bchw', logits.float().softmax(-1)[..., :40], masks.float().sigmoid())


def restore_scores(scores, resized_size, orig_size, input_size=384):
    canvas = (input_size, input_size) if isinstance(input_size, int) else tuple(input_size)
    scores = F.interpolate(scores[None], size=canvas, mode='bilinear', align_corners=False)
    rh, rw = map(int, resized_size)
    scores = scores[:, :, :rh, :rw]
    return F.interpolate(scores, size=tuple(map(int, orig_size)), mode='bilinear', align_corners=False)[0]

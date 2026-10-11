import torch
from torch import nn
import torch.nn.functional as F
from rfdetr import RFDETRSegSmall
from rfdetr.utilities.tensors import NestedTensor
from .semantic_inference import semantic_aggregation


class RFDETRSemantic(nn.Module):
    """RGB-only semantic adapter; upstream stays untouched.

    Preserve the pretrained two-stage reference proposal mechanism. Its COCO
    ranking/box heads are frozen and receive no semantic target or box loss.
    Only the decoder classifier is replaced by a fresh 41-way softmax head.
    Group DETR is reduced to one group: exactly 100 queries in train and eval.
    """
    def __init__(self, checkpoint, num_classes=40, num_queries=100, pretrained=True,
                 freeze_backbone=False, use_depth=False):
        super().__init__()
        assert num_classes == 40 and num_queries == 100
        api = RFDETRSegSmall(pretrain_weights=checkpoint if pretrained else None, device='cpu')
        self.core = api.model.model
        core = self.core
        core.class_embed = nn.Linear(core.class_embed.in_features, 41)
        nn.init.normal_(core.class_embed.weight, std=0.01)
        nn.init.zeros_(core.class_embed.bias)
        core.num_queries = 100
        core.group_detr = core.transformer.group_detr = 1
        for module in core.transformer.modules():
            if hasattr(module, 'group_detr'):
                module.group_detr = 1
        core.query_feat = nn.Embedding.from_pretrained(core.query_feat.weight[:100].detach().clone(), freeze=False)
        core.refpoint_embed = nn.Embedding.from_pretrained(core.refpoint_embed.weight[:100].detach().clone(), freeze=False)
        for name in ('enc_output', 'enc_output_norm', 'enc_out_class_embed', 'enc_out_bbox_embed'):
            modules = getattr(core.transformer, name)
            setattr(core.transformer, name, nn.ModuleList([modules[0]]))
        core.aux_loss = False
        # These modules only determine detached top-k reference proposals.
        for name, p in core.named_parameters():
            if any(part in name for part in ('bbox_embed', 'enc_out_class_embed', 'enc_output')):
                p.requires_grad_(False)
        self.freeze_backbone = freeze_backbone
        self.encoder.requires_grad_(not freeze_backbone)
        self.use_depth = False
        if use_depth:
            self.expand_patch_embed()

    def expand_patch_embed(self):
        """Grow the DINOv2 patch projection 3->4 channels for depth.

        The depth channel weights start at zero, so at initialisation the model
        is functionally identical to the RGB-only baseline (pretraining fully
        preserved) while remaining trainable towards depth cues.
        """
        projection = self.encoder.encoder.embeddings.patch_embeddings.projection
        assert projection.in_channels == 3, 'Patch projection is not 3-channel'
        grown = nn.Conv2d(4, projection.out_channels, kernel_size=projection.kernel_size,
                          stride=projection.stride, padding=projection.padding,
                          bias=projection.bias is not None)
        with torch.no_grad():
            grown.weight.zero_()
            grown.weight[:, :3].copy_(projection.weight)
            grown.bias.copy_(projection.bias)
        patch_embeddings = self.encoder.encoder.embeddings.patch_embeddings
        patch_embeddings.projection = grown
        patch_embeddings.num_channels = 4
        self.use_depth = True

    @property
    def encoder(self):
        return self.core.backbone[0].encoder

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.encoder.eval()
        return self

    def forward(self, images, padding_mask=None, depth=None):
        if self.use_depth:
            assert depth is not None, 'RGB-D model requires a depth input'
            images = torch.cat([images, depth.to(images.dtype)], dim=1)
        else:
            assert depth is None, 'Depth input passed to an RGB-only model'
        if padding_mask is None:
            padding_mask = torch.zeros(images.shape[0], *images.shape[-2:], dtype=torch.bool, device=images.device)
        samples = NestedTensor(images, padding_mask)
        features, positions, cross_features = self.core.backbone(samples)
        srcs, masks = zip(*(f.decompose() for f in features))
        cross_srcs = None if cross_features is None else [f.tensors for f in cross_features]
        hs = self.core.transformer(list(srcs), list(masks), positions,
                                   self.core.refpoint_embed.weight, self.core.query_feat.weight,
                                   cross_attn_srcs=cross_srcs)[0]
        masks = self.core.segmentation_head(features[0].tensors, hs, tuple(images.shape[-2:]))[-1]
        logits = self.core.class_embed(hs[-1])
        scores = semantic_aggregation(logits, masks)
        scores = F.interpolate(scores, size=images.shape[-2:], mode='bilinear', align_corners=False)
        return dict(pred_logits=logits, pred_masks=masks, semantic_logits=scores)

    def parameter_groups(self, lr, backbone_lr):
        encoder_ids = {id(p) for p in self.encoder.parameters()}
        return [dict(params=[p for p in self.parameters() if p.requires_grad and id(p) not in encoder_ids], lr=lr),
                dict(params=[p for p in self.encoder.parameters() if p.requires_grad], lr=backbone_lr)]

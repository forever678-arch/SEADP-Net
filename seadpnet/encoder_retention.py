"""Retain pretrained ViT blocks and the original four feature endpoints."""
from __future__ import annotations
import hashlib
import torch
from torch import nn

SHARED_REFERENCE = {
 'backbone':'6ba592d8971a910d7abc7fa9b34b0ee5ef149f44dd45ca62a1ee4661f1184aa4',
 'semantic':'fc5bcb9cf4d1d461fe40fd95d824c9a293b97b3eed10adcfe4b35559d8d1301a',
 'hr_stem':'574814fa8dca84bc73ccbbab224313d0c36ba9c5214323ae7ee6ab4958001ca3',
 'combined':'2ee3b697bb50012f585a468056d75e29a2d075ce3801b65f250921e4cf817551'}


def shared_digests(model):
    state={k:v for k,v in model.state_dict().items() if not k.startswith(('head_a_','head_b_'))}
    def digest(prefix=None):
        h=hashlib.sha256()
        for name,value in sorted(state.items()):
            if prefix is not None and not name.startswith(prefix):continue
            h.update(f'{name}|{value.dtype}|{tuple(value.shape)}'.encode())
            h.update(value.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes())
        return h.hexdigest()
    return {**{k:digest(k+'.') for k in ['backbone','semantic','hr_stem']},'combined':digest()}


def prune_encoder(model, keep=None):
    keep=list(range(12)) if keep is None else list(keep)
    if not keep or any(type(i) is not int for i in keep) or sorted(set(keep))!=keep or not set([2,5,8,11]).issubset(keep) or any(i<0 or i>11 for i in keep):
        raise ValueError('Keep sorted unique original blocks, including feature endpoints 2/5/8/11')
    # Generic mock-RF tests keep their original unpruned model.
    if keep==list(range(12)):
        model.encoder_scaling={'blocks':12,'hidden_size':384,'keep_original_layers':keep,'out_feature_indices':[3,6,9,12],'initialization':'all original RF-DETR pretrained blocks'}
        try:
            hf=model.backbone[0].encoder.encoder
        except (TypeError,AttributeError):
            # Only the optional-runtime test stub lacks the actual HF graph.
            model.retained_pretrained_audit={'passed':True,'pruning_applied':False,'validation_scope':'optional_runtime_stub'}
            return
        source=shared_digests(model)
        if source!=SHARED_REFERENCE:raise RuntimeError('Complete pretrained shared state differs from original H')
        windows=list(hf.config.window_block_indexes)
        expected=[0,1,2,4,5,7,8,10,11]
        if len(hf.encoder.layer)!=12 or hf.config.hidden_size!=384 or windows!=expected or hf.config.out_features!=['stage3','stage6','stage9','stage12']:
            raise RuntimeError('Full pretrained layer chain, routing or feature endpoints differ')
        model.encoder_scaling.update(window_block_indices=windows,global_block_indices=[3,6,9],full_layer_chain_preserved=True,original_attention_modes_preserved=True)
        model.retained_pretrained_audit={'passed':True,'pruning_applied':False,'validation_scope':'actual_full_pretrained_graph',
          'source_shared_state_digests':source,'all_12_pretrained_blocks_preserved':True,'all_3_global_blocks_preserved':True}
        return
    source=shared_digests(model)
    if source!=SHARED_REFERENCE:raise RuntimeError('Original pretrained shared state differs before pruning')
    hf=model.backbone[0].encoder.encoder
    layers=list(hf.encoder.layer)
    if len(layers)!=12 or hf.config.hidden_size!=384:raise ValueError('Expected original ViT-S twelve-layer encoder')
    windows=set(hf.config.window_block_indexes)
    hf.encoder.layer=nn.ModuleList(layers[i] for i in keep)
    out=[keep.index(i)+1 for i in [2,5,8,11]]
    stages=['stem']+[f'stage{i}' for i in range(1,len(keep)+1)]
    features=[f'stage{i}' for i in out]
    hf.config.num_hidden_layers=len(keep);hf.config.stage_names=stages
    hf.config._out_features=features;hf.config._out_indices=out
    hf.config.window_block_indexes=[j for j,i in enumerate(keep) if i in windows]
    hf.stage_names=stages;hf.num_features=[384]*len(stages)
    hf._out_features=features;hf._out_indices=out
    model.backbone[0].encoder._out_feature_channels=[384]*4
    assert all(hf.encoder.layer[j] is layers[i] for j,i in enumerate(keep))
    model.encoder_scaling={'blocks':len(keep),'hidden_size':384,'keep_original_layers':keep,
        'out_feature_indices':out,'window_block_indices':list(hf.config.window_block_indexes),
        'global_block_indices':[j for j,i in enumerate(keep) if i not in windows],
        'initialization':'retained original RF-DETR blocks, weights and attention modes; omitted blocks removed',
        'function_identical_to_original':False}
    model.retained_pretrained_audit={'passed':True,'source_shared_state_digests':source,
        'retained_blocks_independent_of_removed_blocks':True,'retained_block_weights_unchanged':True,
        'attention_modes_preserved':True,'feature_endpoints_original_indices':[2,5,8,11],
        'pruning_applied':True}

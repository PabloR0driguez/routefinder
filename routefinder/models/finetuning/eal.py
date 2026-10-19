from copy import deepcopy

import torch

from routefinder.models.env_embeddings.mtvrp.context import MTVRPContextEmbeddingFull
from routefinder.models.env_embeddings.mtvrp.init import MTVRPInitEmbeddingFull

from .utils import freeze_backbone


def _copy_projection(source, destination):
    """Copy an existing feature prefix without changing Linear parameter shapes."""
    destination.to(device=source.weight.device, dtype=source.weight.dtype)
    with torch.no_grad():
        destination.weight.zero_()
        destination.weight[:, : source.in_features].copy_(source.weight)
        if source.bias is not None:
            destination.bias.copy_(source.bias)






def efficient_adapter_layers(
    model,
    init_embedding_cls=MTVRPInitEmbeddingFull,
    context_embedding_cls=MTVRPContextEmbeddingFull,
    init_embedding_num_new_feats=1,
    context_embedding_num_new_feats=3,
    adapter_only=False,
):
    """Efficient Active Layers.
    Keep the model the same, replace the embeddings with new zero-padded
    embeddings for unseen features.

    Args:
        model: the model to be adapted
        init_embedding_cls: the new init embedding class
        context_embedding_cls: the new context embedding class
        init_embedding_num_new_feats: the number of new features added to
        the init embedding, initialized with zeros
        context_embedding_num_new_feats: the number of new features added
        to the context embedding, initialized with zeros
        adapter_only: if True, only the modified embedding modules are trained;
        otherwise, the whole model is trained
    """

    print("Using Efficient Adapter Layers (EAL)")

    policy = model.policy
    embed_dim = policy.decoder.context_embedding.embed_dim
    policy_device = next(policy.parameters()).device

    # Copy the pretrained policy before modifying its embeddings
    policy_new = deepcopy(policy)

    # Init Embedding with EAL (encoder)
    # Only modify the encoder embedding when a new class is provided
    if init_embedding_cls is not None:
        init_embedding_old = deepcopy(policy.encoder.init_embedding)

        # use previous k weights and pad l new number of weights with zeros
        proj_glob_params_old = (
            init_embedding_old.project_global_feats.weight.data)


        init_embed_new = init_embedding_cls(
            embed_dim=embed_dim
        )
        _copy_projection(init_embedding_old.project_global_feats, init_embed_new.project_global_feats,)
        _copy_projection(init_embedding_old.project_customers_feats, init_embed_new.project_customers_feats,)

 
        # Replace the encoder embedding only when it was modified
        policy_new.encoder.init_embedding = init_embed_new

    # Context Embedding with EAL (decoder)
    if context_embedding_cls is not None:
        context_embedding_old = deepcopy( policy.decoder.context_embedding)

        context_embed_new = context_embedding_cls(
            embed_dim=embed_dim
        )
        _copy_projection(context_embedding_old.project_context, context_embed_new.project_context,)

        policy_new.decoder.context_embedding = context_embed_new
        


    # Move the modified policy to the original policy device
    policy_new = policy_new.to(policy_device)

    # If adapter_only is enabled, freeze the backbone
    if adapter_only:
        # Train the encoder initial embedding only if we modified it
        train_init_embedding = init_embedding_cls is not None
        train_context_embedding = context_embedding_cls is not None 
        policy_new = freeze_backbone(
            policy_new,
            train_everything=train_init_embedding,
            train_context_embedding=train_context_embedding,
        )


    model.policy = policy_new
    return model
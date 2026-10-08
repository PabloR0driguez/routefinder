from copy import deepcopy

import torch

from routefinder.models.env_embeddings.mtvrp.context import MTVRPContextEmbeddingFull
from routefinder.models.env_embeddings.mtvrp.init import MTVRPInitEmbeddingFull

from .utils import freeze_backbone


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
        init_embedding_old = deepcopy(
            policy.encoder.init_embedding
        )

        # use previous k weights and pad l new number of weights with zeros
        proj_glob_params_old = (
            init_embedding_old.project_global_feats.weight.data
        )
        proj_glob_params_new = torch.cat(
            [
                proj_glob_params_old,
                torch.zeros(
                    (
                        proj_glob_params_old.shape[0],
                        init_embedding_num_new_feats,
                    ),
                    device=proj_glob_params_old.device,
                    dtype=proj_glob_params_old.dtype,
                ),
            ],
            dim=-1,
        )

        init_embed_new = init_embedding_cls(
            embed_dim=embed_dim
        )
        init_embed_new.project_global_feats.weight.data = (
            proj_glob_params_new
        )
        init_embed_new.project_customers_feats.weight.data = (
            init_embedding_old.project_customers_feats.weight.data
        )

        # Replace the encoder embedding only when it was modified
        policy_new.encoder.init_embedding = init_embed_new

    # Context Embedding with EAL (decoder)
    if context_embedding_cls is not None:
        context_embedding_old = deepcopy(
            policy.decoder.context_embedding
        )

        # use previous k weights and pad l new number of weights with zeros
        proj_context_old = (
            context_embedding_old.project_context.weight.data
        )
        proj_context_new = torch.cat(
            [
                proj_context_old,
                torch.zeros(
                    (
                        proj_context_old.shape[0],
                        context_embedding_num_new_feats,
                    ),
                    device=proj_context_old.device,
                    dtype=proj_context_old.dtype,
                ),
            ],
            dim=-1,
        )

        # Replace the context embedding with the new one
        # that contains vehicle capacity and speed
        context_embed_new = context_embedding_cls(
            embed_dim=embed_dim
        )
        context_embed_new.project_context.weight.data = (
            proj_context_new
        )

        policy_new.decoder.context_embedding = (
            context_embed_new
        )

    # Move the modified policy to the original policy device
    policy_new = policy_new.to(policy_device)

    # If adapter_only is enabled, freeze the backbone
    if adapter_only:
        # Train the encoder initial embedding only if we modified it
        train_init_embedding = init_embedding_cls is not None

        policy_new = freeze_backbone(
            policy_new,
            train_everything=train_init_embedding,
        )

    model.policy = policy_new

    return model
def freeze_backbone(policy, train_everything=True,):
    # Freeze all the parameters in the model
    for param in policy.parameters():
        param.requires_grad = False
    # Unfreeze embeddings
    #now that we want to add features only on context
    #ie only on the decoder so we need to modify this
    #but lets keep default behavior untouched
    if train_everything:
        for param in policy.encoder.init_embedding.parameters():
            param.requires_grad = True
    for param in policy.decoder.context_embedding.parameters():
        param.requires_grad = True
    return policy
#now we go to eal.py

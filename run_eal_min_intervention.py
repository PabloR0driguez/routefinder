import os
from unittest.mock import patch

import torch

import torchrl.data.tensor_specs as tensor_specs



#next imports are for naming compatibility
from torchrl.data import Bounded,  Composite, UnboundedContinuous,UnboundedDiscrete

tensor_specs.CompositeSpec = Composite
tensor_specs.BoundedTensorSpec = Bounded
tensor_specs.UnboundedContinuousTensorSpec = UnboundedContinuous
tensor_specs.UnboundedDiscreteTensorSpec = UnboundedDiscrete



import wandb

from lightning.pytorch.callbacks import LearningRateMonitor, RichModelSummary
from lightning.pytorch.loggers import WandbLogger
from rl4co.utils.callbacks.speed_monitor import SpeedMonitor

# new model
from rl4co.utils.trainer import RL4COTrainer

#from routefinder.envs.mtdvrp import MTVRPEnv, MTVRPGenerator
from routefinder.models.baselines.mvmoe.model import MVMoE
#from routefinder.models.env_embeddings.mtvrp.context import MTVRPContextEmbeddingFull
#from routefinder.models.env_embeddings.mtvrp.init import MTVRPInitEmbeddingFull
from routefinder.models.finetuning.baselines import adapter_layers, model_from_scratch
from routefinder.models.finetuning.eal import efficient_adapter_layers

## Normal training (note that we will actually just load a checkpoint)
## Zero shot (after training)
from routefinder.models.model import (
    RouteFinderBase,
    RouteFinderMoE,
    RouteFinderSingleVariantSampling,
)


from routefinder.models.env_embeddings.mtvrp.context import MTVRPContextEmbeddingExternal
from routefinder.envs.mtvrp import MTVRPEnv, MTVRPGenerator


# Load data into env
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# We have 3 different EAL settings: MB (mixed backhaul), MD (multi-depot), and both
VARIANT_GEN_SETTINGS = {
    "mb": {"backhaul_ratio": 0.3, "sample_backhaul_class": True, "num_depots": 1},
    "md": {"backhaul_ratio": 0.2, "sample_backhaul_class": False, "num_depots": 3},
    "both": {"backhaul_ratio": 0.3, "sample_backhaul_class": True, "num_depots": 3},
}


def finetune_variant_names(variant_name):
    # 3 cases: finetune_var = "mb", "md", "both"
    variant_names = []
    for root, dirs, files in os.walk("data"):
        # get dir whose name contains "finetune_variant"
        for dir in dirs:
            if "md" not in dir and "mb" not in dir:
                continue
            if variant_name == "mb":
                if "md" in dir:
                    continue
            elif variant_name == "md":
                if "mb" in dir:
                    continue
            else:
                if not ("mb" in dir and "md" in dir):
                    continue
            variant_names.append(dir)
    return variant_names


# Load checkpoint

# Before loading checkpoint we need to modify the loader
# because the checkpoint has its own old MTVRPGenerator but init is not triggered
# but our new _generate() DOES have vehicle_data while the saved generator doesnt
# model.setup() tries to generate the train data with the env inside the checkpoint



def load_from_checkpoint_no_setup(
    model_cls,
    checkpoint_path,
):
    # what are we using? what are our parameters?
    # Load the pretrained model parameters, but do not initialize
    # the REINFORCE baseline with the old environment from the checkpoint
    model = model_cls.load_from_checkpoint(
        checkpoint_path,
        map_location="cpu",
        strict=False,
        weights_only=False,
        load_baseline=False,
    )
    # load model with our params
    # The new environment and datasets will be initialized later
    return model




def main(
    path,
    model_type="rf",
    train_type="eal-adapter",
    finetune_variant="vehicles",
    lr=3e-4,
    test_zero_shot=False,
):
    if "rf" in model_type:
        if "moe" in model_type:
            model = load_from_checkpoint_no_setup(RouteFinderMoE, path,)
        else:
            #AttributeError: 'MTVRPGenerator' object has no attribute 'vehicle_generator'
            # how to solve?
            #model = RouteFinderBase.load_from_checkpoint(path, map_location="cpu", weights_only=False,)
            model = load_from_checkpoint_no_setup(RouteFinderBase, path,)
            model = model.to(device)

    elif model_type == "mvmoe":
        model = load_from_checkpoint_no_setup(MVMoE, path,)
    elif model_type == "mtpomo":
        model = load_from_checkpoint_no_setup(RouteFinderSingleVariantSampling, path,)
    else:
        raise ValueError("Model type not recognized: {}".format(model_type))

    model = model.to(device)

    if "eal" in train_type:
        model = efficient_adapter_layers(
            model,
            init_embedding_cls=None,
            context_embedding_cls=MTVRPContextEmbeddingExternal,
            init_embedding_num_new_feats=0,
            context_embedding_num_new_feats=2,
            adapter_only="adapter" in train_type,
        )
        context_shape = (model.policy.decoder.context_embedding.project_context.weight.shape)
        print( "Context shape:",  context_shape,)
    else:
        raise ValueError(
            "Training type not recognized: {}. Choose from ['eal', 'al', 'scratch']".format(
                train_type
            )
        )


    # Create env: the new setting is with backhaul sampling (so we have the new MB variants)
    # and also we have slightly more backhauls
    #Shall we try the minimum intervention setting with all congurations? 
    # Or only the most basic CVRP?  

    generator = MTVRPGenerator(
        num_loc=20,
        variant_preset="cvrp",
        fleet_size=20,
    )

    env = MTVRPEnv(
        generator=generator,
        check_solution=False,
    )


    # Reset learning rate
    model.optimizer_kwargs["lr"] = lr

    # Test model
    model.env = env
    model.data_cfg["batch_size"] = 16
    model.data_cfg["val_batch_size"] = 16
    model.data_cfg["test_batch_size"] = 256
    model.data_cfg["train_data_size"] = 256  # instead of 100k
    model.data_cfg["val_data_size"] = 64
    model.data_cfg["test_data_size"] = 64
    model.setup()


    # Test model
    logger = False
    rich_model_summary = RichModelSummary(max_depth=3)
    speed_monitor = SpeedMonitor(
        intra_step_time=True, inter_step_time=True, epoch_time=True
    )
    #lr_monitor = LearningRateMonitor(logging_interval="epoch")

    max_epochs = 10
    trainer = RL4COTrainer(
        accelerator="auto",
        devices=1,
        max_epochs=max_epochs,
        logger=logger,
        #callbacks=[rich_model_summary, speed_monitor, lr_monitor],
        callbacks=[rich_model_summary, speed_monitor],

    )

    # Validate and test zero-shot generalization reporting
    trainer.validate(model)
    if test_zero_shot:
        print("Testing zero-shot generalization")
        log_dict = model.log_dict

        def log_zero_shot_metrics(metrics, *args, **kwargs):
            metrics = {
                ("zero_shot/" + name.removeprefix("test/"))
                if name.startswith("test/") else name: value
                for name, value in metrics.items()
            }
            return log_dict(metrics, *args, **kwargs)

        with patch.object(model, "log_dict", log_zero_shot_metrics):
            #trainer.test(model)
            test_results = trainer.test(model)
            test_cost = -test_results[0]["test/max_aug_reward"]
            print( "test cost:",  test_cost,)

    # Main training loop
    trainer.fit(model)

    # Test model
    trainer.test(model)

    print("Finished training")
    #wandb.finish()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type", type=str, default="rf", help="Model type: rf, mvmoe, mtpomo"
    )
    parser.add_argument("--experiment", type=str, default="eal-adapter", help="Experiment type")
    parser.add_argument(
        "--variants_finetune", type=str, default="vehicles", help="Variants to finetune on"
    )
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/100/rf-transformer.ckpt"
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num_runs", type=int, default=3)

    args = parser.parse_args()

    # available_experiments = ["eal-full", "eal-adapter", "al-full", "al-adapter", "scratch"]
    available_experiments = ["eal-adapter"]
    if args.experiment == "all":
        exps = available_experiments
    else:
        exps = [args.experiment]
        assert all(
            [e in available_experiments for e in exps]
        ), f"Invalid experiment: {exps}. Choose from {available_experiments}"

    #available_variants = ["mb", "md", "both"]  # mixed backhaul, multi-depot, both
    available_variants = ["vehicles"]
    if args.variants_finetune == "all":
        variants = available_variants
    else:
        variants = [args.variants_finetune]
        assert all(
            [v in available_variants for v in variants]
        ), f"Invalid variant: {variants}. Choose from {available_variants}"

    for finetune_variant in variants:
        print(f"Finetuning on {finetune_variant} variants")
        for exp in exps:
            print(f"Training for {exp}")
            for i in range(args.num_runs):
                print(f"Run {i+1}/{args.num_runs}")
                main(args.checkpoint, args.model_type, exp, finetune_variant, args.lr)

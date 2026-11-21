


import os
from unittest.mock import patch
import time

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


#from routefinder.models.env_embeddings.mtvrp.context import MTVRPContextEmbeddingExternal
from routefinder.models.env_embeddings.mtvrp.init import MTVRPInitEmbeddingDual
from routefinder.envs.mtvrp import MTVRPEnv, MTVRPGenerator

#new imports to save results
import csv
import pickle


#import also some functions from run_fleet_size_only.py so to not re do them
from rl4co.data.utils import load_npz_to_tensordict
from routefinder.data.utils import get_dataloader
from run_fleet_size_only import test, count_vehicles_used

# IMPORTANT NEW IMPORT
from routefinder.models.env_embeddings.mtvrp.context import MTVRPContextEmbeddingCurrentVehicle
from rl4co.utils.ops import unbatchify

# Load data into env
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    datasets=None,
    fleet_size=100,
    num_instances=100,
    eval_batch_size=4,
    run_id=1,
    save_results=True,
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
    print("Loading dataset...", flush=True)
    problem = os.path.basename(datasets).split("_")[0].lower()
    td_test = load_npz_to_tensordict(datasets)[:num_instances]
    location_amount=td_test["locs"].shape[-2] - 1
    #locs has the instances locations, -2 is the second to last dim aka node count, -1 deletes depot
    generator = MTVRPGenerator(
        num_loc= location_amount,
        variant_preset=problem,
        fleet_size=  td_test["vehicle_capacities"].shape[-1],
        fleet_mode="two_type",
        small_vehicle_capacity=20.0,
        large_vehicle_capacity=40.0,
        small_vehicle_speed=0.5,
        large_vehicle_speed=1.0,
    )


    model = efficient_adapter_layers(
        model,
        init_embedding_cls=MTVRPInitEmbeddingDual,
        context_embedding_cls=MTVRPContextEmbeddingCurrentVehicle,
        init_embedding_num_new_feats=6,
        context_embedding_num_new_feats=2,
        adapter_only=True,
        new_features_only=True,
    )

    #ADD optimizer for fine tuning

    context_weight = (model.policy.decoder.context_embedding.project_context.weight)
    optimizer = torch.optim.Adam([context_weight],  lr=1e-3,  weight_decay=0.0, ) #lr=3e-4 was not showing improvements

    # Evaluate the saved instances
    policy = model.policy.to(device).eval()
    results = {}
    env_constrained = MTVRPEnv(generator=generator,check_solution=False,   enforce_fleet_limit=True,  handle_dead_ends=True,)
    #
    env_unconstrained = MTVRPEnv(generator=generator, check_solution=False, enforce_fleet_limit=True, handle_dead_ends=True,)
    
    # I AM HERE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # Use fresh training instances, keeping the saved test instances for evaluation.
    # Seed once; later this will stay outside the training loop.
    torch.manual_seed(123)

    # Start with a small batch and eight sampled starts to check the training setup.
    train_num_starts = 8
    training_steps = 500

    for step in range(1, training_steps + 1):

        train_batch = generator([16])

        # Apply the same vehicle-prefix selection used during evaluation.
        for key in ("vehicle_capacities", "vehicle_speeds"):
            train_batch[key] = train_batch[key][:, :fleet_size].clone()

        train_batch["num_vehicles"] = torch.full_like(
            train_batch["num_vehicles"], fleet_size
        )

        td_train = env_constrained.reset(train_batch).to(device)

        train_output = policy(
            td_train,
            env_constrained,
            phase="train",
            decode_type="multistart_sampling",
            num_starts=train_num_starts,
            select_best=False,
            return_actions=False,
        )
        #Now how to reward?
        #What to do with an unfeasible solution

        details = env_constrained.evaluation_result
        route_cost = details["route_cost"]
        violation = details["unserved_customers"]
        per_cust_penalty= 2.0 #2 abitrary
        reward = -(route_cost + per_cust_penalty* violation)

        reward = unbatchify(reward, train_num_starts)
        log_likelihood = unbatchify(  train_output["log_likelihood"], train_num_starts)

        # Use the average reward of each instance as its baseline
        baseline = reward.mean(dim=-1, keepdim=True)

        # Positive advantage means this route performed better than its average.
        # Treat this comparison as fixed feedback during gradient calculation.
        advantage = (reward - baseline).detach()

        # REINFORCE loss encourages better-than-average sampled routes.
        loss = -(advantage * log_likelihood).mean()
        torch.nn.utils.clip_grad_norm_([context_weight], 1.0)
        
        print(
            f"Update {step}/{training_steps} | Loss: {loss.item():.4f}",
            flush=True,
        )
        # Stop before updating weights if the loss is invalid.
        if not torch.isfinite(loss).item():
            raise RuntimeError("Training loss is not finite")

        # Save the existing columns so we can verify they remain unchanged.
        old_weights = context_weight[:, :-2].detach().clone()

        optimizer.zero_grad()  # Clear any previous gradients.
        loss.backward()       # Calculate gradients; our hook zeros the old columns.
        grad_norm = torch.nn.utils.clip_grad_norm_([context_weight], 1.0)
        optimizer.step()      # Update the two new EAL columns.

        feasible_rate = (violation == 0).float().mean().item()
        print(
            f"Update {step}/{training_steps} | cost {route_cost.mean():.3f} | "
            f"unserved {violation.float().mean():.3f} | feasible {feasible_rate:.2%} | "
            f"grad {grad_norm:.3f}",
            flush=True,
        )
        print(
            "Old columns unchanged:",
            torch.equal(old_weights, context_weight[:, :-2].detach()),
            flush=True,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="rf")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/100/rf-transformer.ckpt",
    )
    parser.add_argument("--datasets", type=str, required=True)
    parser.add_argument("--fleet_size", type=int, default=100)
    parser.add_argument("--num_instances", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument(
        "--save-results",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    for i in range(args.num_runs):
        print(f"Evaluation run {i + 1}/{args.num_runs}", flush=True)

        main(
            path=args.checkpoint,
            model_type=args.model_type,
            datasets=args.datasets,
            fleet_size=args.fleet_size,
            num_instances=args.num_instances,
            eval_batch_size=args.batch_size,
            run_id=i + 1,
            save_results=args.save_results,
        )



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
        fleet_size=  fleet_size,
        fleet_mode="two_type",
        small_vehicle_capacity=20.0,
        large_vehicle_capacity=40.0,
        small_vehicle_speed=0.5,
        large_vehicle_speed=1.0,
    )


    model = efficient_adapter_layers(
        model,
        init_embedding_cls=MTVRPInitEmbeddingDual,
        context_embedding_cls=None,
        init_embedding_num_new_feats=6,
        context_embedding_num_new_feats=0,
        adapter_only=True,
    )


    # Evaluate the saved instances
    policy = model.policy.to(device).eval()
    results = {}
    env_constrained = MTVRPEnv(generator=generator,check_solution=False,   enforce_fleet_limit=True,  handle_dead_ends=True,)
    #
    env_unconstrained = MTVRPEnv(generator=generator, check_solution=False, enforce_fleet_limit=True, handle_dead_ends=True,)
    td_all = td_test.clone()
    num_customers = td_all["locs"].shape[-2] - 1
    stored_fleet_size = td_all["vehicle_capacities"].shape[-1]
    for key in ("vehicle_capacities", "vehicle_speeds"):
        td_all[key] = td_all[key][:, :fleet_size].clone()
    td_all["num_vehicles"] = torch.full_like(td_all["num_vehicles"], fleet_size)

    # On a test no solution was found, this mini section will eval capacities
    capacities = td_all["vehicle_capacities"]
    print("Total demand:",
          td_all["demand_linehaul"].sum(-1).tolist(), flush=True)
    print("Total fleet capacity:",
          capacities.sum(-1).tolist(), flush=True)
    print("Both capacity types present:",
          (capacities.min(dim=-1).values
           < capacities.max(dim=-1).values).tolist(), flush=True)
    #-------
    reference_size = max(num_customers, fleet_size)
    repeats = (reference_size + fleet_size - 1) // fleet_size

    dataloader = get_dataloader(td_all, batch_size=eval_batch_size)
    start = time.time()
    unconstrained_results, constrained_results, vehicle_counts = [], [], []

    processed_instances = 0
    total_instances = td_all.batch_size[0]


    for batch in dataloader:
        print("Starting batch...", flush=True)
        source_batch = batch.clone()

        # Keep an untouched initial state for the correctness check.
        td_check_con = env_constrained.reset(source_batch.clone()).to(device)
        td_con = td_check_con.clone()

        print(
            f"Instances {processed_instances + 1}–"f"{processed_instances + len(batch)} of {total_instances}",
            flush=True,)

        o = test(policy, td_con, env_constrained, num_augment=8, num_starts=None, device=device,)

        # Validate.
        solved = o["has_solution"]
        if solved.any().item():
            env_constrained.check_solution_validity(
                td_check_con[solved], o["best_aug_actions"][solved]
            )
        constrained_results.append(o)

        # UNCONSTRAINED: repeat the whole fleet instead of vehicle 0.
        unconstrained_batch = source_batch.clone()
        for key in ("vehicle_capacities", "vehicle_speeds"):
            unconstrained_batch.set(
                key,
                source_batch[key]
                .repeat(1, repeats)[:, :reference_size]
                .clone(),
            )
        unconstrained_batch.set(
            "num_vehicles",
            torch.full_like(source_batch["num_vehicles"], reference_size),
        )

        td_check_uncon = env_unconstrained.reset(unconstrained_batch).to(device)
        td_uncon = td_check_uncon.clone()

        o_uncon = test(policy, td_uncon, env_unconstrained,num_augment=8, num_starts=None, device=device,)

        solved = o_uncon["has_solution"]
        if solved.any().item():
            env_unconstrained.check_solution_validity(
                td_check_uncon[solved],
                o_uncon["best_aug_actions"][solved],
            )
        unconstrained_results.append(o_uncon)

        vehicles_used = count_vehicles_used(o_uncon["best_aug_actions"])
        print("Vehicles used:", vehicles_used.cpu().tolist(), flush=True)
        vehicle_counts.append(vehicles_used)

        processed_instances += len(batch)
        print(
            f"Completed {processed_instances}/{total_instances}",
            flush=True,
        )

    out = {}
    vehicle_counts = torch.cat(vehicle_counts)
    unconstrained_rewards = torch.cat([out["max_aug_reward"] for out in unconstrained_results])
    constrained_rewards = torch.cat([out["max_aug_reward"] for out in constrained_results])

    unconstrained_solved = torch.cat([out["has_solution"] for out in unconstrained_results])
    constrained_solved = torch.cat([out["has_solution"] for out in constrained_results]) 
    num_instances = constrained_solved.numel()
    constrained_successes = int(constrained_solved.sum().item())
    unconstrained_successes = int(unconstrained_solved.sum().item())

    exceeded_per_instance= (vehicle_counts[unconstrained_solved] > fleet_size).float().mean().item()  if unconstrained_successes else None#bools need to be float before mean
    constrained_cost= -constrained_rewards[constrained_solved].mean().item()  if constrained_successes else None
    unconstrained_cost= -unconstrained_rewards[unconstrained_solved].mean().item()   if unconstrained_successes else None
    constrained_cost_both = unconstrained_cost_both = masking_penalty = None
    common_solved = constrained_solved & unconstrained_solved
    #we need common_solved and the two paired averages
    #None when there are no shared successes or the reference cost is zero, because the percentage is undefined.
    if common_solved.any().item():
        constrained_cost_both= -constrained_rewards[common_solved].mean().item()
        unconstrained_cost_both= -unconstrained_rewards[common_solved].mean().item()
        if unconstrained_cost_both != 0:
            masking_penalty = (constrained_cost_both - unconstrained_cost_both) / unconstrained_cost_both
    #masking_penalty =  (constrained_cost - unconstrained_cost) / unconstrained_cost
    inference_time = time.time() - start
    num_instances = sum(out["has_solution"].numel() for out in constrained_results)
    constrained_successes = sum(int(out["has_solution"].sum().item()) for out in constrained_results)
    unconstrained_successes = sum(int(out["has_solution"].sum().item()) for out in unconstrained_results)


    print(
        f"DUAL_FLEET_{problem.upper()} | "
        f"Budget: {fleet_size} | "
        f"Exceed rate: {f'{exceeded_per_instance:.2%}' if exceeded_per_instance is not None else 'n/a'} | "
        f"Failures: constrained {num_instances - constrained_successes}, unconstrained {num_instances - unconstrained_successes} | "
        f"Masking penalty: {f'{masking_penalty:.2%}' if masking_penalty is not None else 'n/a'} | "
        f"Evaluation time: {inference_time:.3f} s"
    )

    dataset_name = f"DUAL_FLEET_{problem.upper()}"
    results[dataset_name] = {
        "cost": constrained_cost,
        "unconstrained_cost": unconstrained_cost,
        "exceedance_rate": exceeded_per_instance,
        "masking_penalty": masking_penalty,
        "constrained_failures": num_instances - constrained_successes,
        "unconstrained_failures": num_instances - unconstrained_successes,
        "evaluation_time": inference_time,
    }


    #unlike homogeneous, will keep the foresight to save as csv as we run
    if save_results:
        stage = "zero_shot"

        row = {
            "problem": problem.upper(), "stage": stage,
            "run_id": run_id,
            "fleet_budget": fleet_size, "instances": num_instances,
            "violations": int((vehicle_counts[unconstrained_solved] > fleet_size).sum().item()),
            "unconstrained_solved": unconstrained_successes,
            "constrained_solved": constrained_successes,
            "paired_instances": int(common_solved.sum().item()),
            "paired_constrained_cost": constrained_cost_both,
            "paired_unconstrained_cost": unconstrained_cost_both,
            "masking_penalty_pct": (100 * masking_penalty if masking_penalty is not None else None),
            "evaluation_time": inference_time,
            "dataset": datasets,
            "checkpoint": path,
        }
        results[dataset_name].update(row)

        # Save dual results separately from the homogeneous experiment.
        savedir = f"results/dual/{location_amount}/"
        os.makedirs(savedir, exist_ok=True)

        checkpoint_name = os.path.splitext(os.path.basename(path))[0]
        filename = (
            f"{checkpoint_name}_{problem}_{stage}"
            f"_fleet{fleet_size}_n{num_instances}_run{run_id}.pkl"
        )

        with open(savedir + filename, "wb") as file:
            pickle.dump(results, file)

        with open(
            savedir + "dual_fleet_results.csv",
            "a", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(file, fieldnames=list(row))
            if file.tell() == 0:
                writer.writeheader()
            writer.writerow(row)

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
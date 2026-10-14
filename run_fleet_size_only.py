import argparse
import os
import pickle
import time
import warnings

import torch

import matplotlib.pyplot as plt


from rl4co.data.transforms import StateAugmentation
from rl4co.utils.ops import gather_by_index, unbatchify
from tqdm.auto import tqdm

from routefinder.data.utils import get_dataloader
from routefinder.envs import MTVRPEnv
# ---> New imports
from routefinder.envs.mtvrp.generator import MTVRPGenerator
# <---

#New imports added for compatibility
import torchrl.data.tensor_specs as tensor_specs
from torchrl.data import (Bounded, Composite, UnboundedContinuous, UnboundedDiscrete,)
tensor_specs.CompositeSpec = Composite
tensor_specs.BoundedTensorSpec = Bounded
tensor_specs.UnboundedContinuousTensorSpec = UnboundedContinuous
tensor_specs.UnboundedDiscreteTensorSpec = UnboundedDiscrete

#End of new imports
from routefinder.models import RouteFinderBase, RouteFinderMoE
from routefinder.models.baselines.mtpomo import MTPOMO
from routefinder.models.baselines.mvmoe import MVMoE

# Tricks for faster inference
try:
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_set_profiling_mode(False)
except AttributeError:
    pass

torch.set_float32_matmul_precision("medium")


def test(
    policy,
    td,
    env,
    num_augment=8,
    augment_fn="dihedral8",  # or symmetric. Default is dihedral8 for reported eval
    num_starts=None,
    device="cuda",
):

    costs_bks = td.get("costs_bks", None)

    with torch.inference_mode():
        with (
            torch.amp.autocast("cuda")
            if "cuda" in str(device)
            else torch.inference_mode()
        ):  # Use mixed precision if supported
            n_start = env.get_num_starts(td) if num_starts is None else num_starts

            if num_augment > 1:
                td = StateAugmentation(num_augment=num_augment, augment_fn=augment_fn)(td)

            # Evaluate policy
            out = policy(td, env, phase="test", num_starts=n_start, return_actions=True)

            # Unbatchify reward to [batch_size, num_augment, num_starts].
            reward = unbatchify(out["reward"], (num_augment, n_start))
            print("[DEBUG] Reshaped reward:", reward.shape)
            if n_start > 1:
                # max multi-start reward
                max_reward, max_idxs = reward.max(dim=-1)
                out.update({"max_reward": max_reward})

                if out.get("actions", None) is not None:
                    # Reshape batch to [batch_size, num_augment, num_starts, ...]
                    actions = unbatchify(out["actions"], (num_augment, n_start))
                    out.update(
                        {
                            "best_multistart_actions": gather_by_index(
                                actions, max_idxs, dim=max_idxs.dim()
                            )
                        }
                    )
                    out["actions"] = actions

            # Get augmentation score only during inference
            if num_augment > 1:
                # If multistart is enabled, we use the best multistart rewards
                reward_ = max_reward if n_start > 1 else reward
                max_aug_reward, max_idxs = reward_.max(dim=1)
                print("[DEBUG] Augmentation selection")
                print(" reward_:", reward_.shape)
                print("  max_idxs:", max_idxs.shape)

                out.update({"max_aug_reward": max_aug_reward})

                # If costs_bks is available, we calculate the gap to BKS
                if costs_bks is not None:
                    # note: torch.abs is here as a temporary fix, since we forgot to
                    # convert rewards to costs. Does not affect the results.
                    gap_to_bks = (
                        100
                        * (-max_aug_reward - torch.abs(costs_bks))
                        / torch.abs(costs_bks)
                    )
                    out.update({"gap_to_bks": gap_to_bks})

                if out.get("actions", None) is not None:
                    actions_ = (
                        out["best_multistart_actions"] if n_start > 1 else out["actions"]
                    )
                    out.update({"best_aug_actions": gather_by_index(actions_, max_idxs)})

            if out.get("gap_to_bks", None) is None:
                out.update({"gap_to_bks": 69420})  # Dummy value

            return out


def count_vehicles_used(actions):
    # Convert actions to a numpy array for easier processing
    
    previous_actions = torch.cat(  (  torch.zeros_like(actions[:, :1]),  actions[:, :-1], ),  dim=1,  )

    new_vehicle = (  (previous_actions == 0)  & (actions != 0)   )
    vehicle_counts = new_vehicle.sum(dim=1)
    return vehicle_counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to the model checkpoint"
    )
    parser.add_argument(
        "--problem",
        type=str,
        default="all",
        help="Problem name: cvrp, vrptw, etc. or all",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=100,
        help="Problem size: 50, 100, for automatic loading",
    )
    parser.add_argument(
        "--datasets",
        help="Filename of the dataset(s) to evaluate. Defaults to all under data/{problem}/ dir",
        default=None,
    )
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--remove-mixed-backhaul",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove mixed backhaul instances. Use --no-remove-mixed-backhaul to keep them.",
    )
    parser.add_argument(
        "--save-results",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save results to results/main/{size}/{checkpoint",
    )
    # ---> New parses for new attributes
    parser.add_argument("--fleet_size", type=int, default=5)
    parser.add_argument("--min_speed", type=float, default=0.5)
    parser.add_argument("--max_speed", type=float, default=1.5)
    parser.add_argument("--min_capacity_factor", type=float, default=0.5)
    parser.add_argument("--max_capacity_factor", type=float, default=1.5)
    parser.add_argument("--num_instances", type=int, default=1000, help="Number of synthetic instances to generate")
    #parser.add_argument("--num_depots",type=int, default=1,)
    # <---
    # Use load_from_checkpoint with map_location, which is handled internally by Lightning
    # Suppress FutureWarnings related to torch.load and weights_only
    warnings.filterwarnings("ignore", message=".*weights_only.*", category=FutureWarning)

    opts = parser.parse_args()

    if "cuda" in opts.device and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # Load model
    print("Loading checkpoint from ", opts.checkpoint)
    if "mvmoe" in opts.checkpoint:
        BaseLitModule = MVMoE
    elif "mtpomo" in opts.checkpoint:
        BaseLitModule = MTPOMO
    elif "moe" in opts.checkpoint:
        BaseLitModule = RouteFinderMoE
    else:
        BaseLitModule = RouteFinderBase

    #Section commented for compatibility of versions purposes
    #model = BaseLitModule.load_from_checkpoint(
    #    opts.checkpoint, map_location="cpu", strict=False
    #)

    #model = BaseLitModule.load_from_checkpoint(
    #    opts.checkpoint, map_location="cpu", strict=False, 
    #    weights_only=False,
    #)


    model = BaseLitModule.load_from_checkpoint(
        opts.checkpoint,
        map_location="cpu",
        strict=False,
        weights_only=False,
        env=MTVRPEnv(),
    )
    # ---> Change in env because now we need to generate our data
    #env = MTVRPEnv()

    problems = [
        "cvrp", "ovrp", "vrpb", "vrpl",
        "ovrpb", "ovrpl", "vrpbl", "vrptw",
        "ovrpbl", "ovrptw", "vrpbtw", "vrpltw",
        "ovrpbtw", "ovrpltw", "vrpbltw", "ovrpbltw",
    ] if opts.problem == "all" else [opts.problem]


    policy = model.policy.to(device).eval()
    results = {}
    for problem in problems:
        print(f"\nEval {problem.upper()}")

        #Fleet size alone with a homogeneous fleet. 
        #No new attributes, existing instances: 
        #measure how often the vehicle budget is exceeded zero-shot 
        #and what pure masking costs in solution quality. 
        generator = MTVRPGenerator(
            num_loc=opts.size,
            variant_preset=opts.problem if opts.problem != "all" else "all",
            fleet_size=opts.fleet_size,
            fleet_mode="homogeneous",
        )
        env_constrained = MTVRPEnv(generator=generator, check_solution=False, enforce_fleet_limit=True,)
        env_unconstrained = MTVRPEnv(generator=generator, check_solution=False, enforce_fleet_limit=False,)

        # random seed
        torch.manual_seed(123)
        td_all = generator(batch_size=[opts.num_instances])
        dataloader = get_dataloader(td_all, batch_size=opts.batch_size)
        start = time.time()
        #res = []
        unconstrained_results, constrained_results, vehicle_counts = [], [], []

        for batch in dataloader:
            source_batch = batch.clone()

            # Correctness check
            td_check_con = env_constrained.reset(source_batch.clone()).to(device)
            o_check = test(policy, td_check_con, env_constrained, num_augment=1, num_starts=1, device=device)
            env_constrained.check_solution_validity(td_check_con, o_check["actions"])

            # Real evaluation
            td_con = env_constrained.reset(source_batch.clone()).to(device)
            o = test(policy, td_con, env_constrained, num_augment=8, num_starts=None, device=device)
            constrained_results.append(o)

            td_uncon = env_unconstrained.reset(source_batch.clone()).to(device)
            o_uncon = test(policy, td_uncon, env_unconstrained, num_augment=8, num_starts=None, device=device)
            unconstrained_results.append(o_uncon)

            vehicles_used = count_vehicles_used(
                o_uncon["best_aug_actions"]
                )
            vehicle_counts.append(vehicles_used.cpu())


        out = {}
        vehicle_counts = torch.cat(vehicle_counts)
        unconstrained_rewards = torch.cat(
            [out["max_aug_reward"] for out in unconstrained_results])
        constrained_rewards = torch.cat(
            [out["max_aug_reward"] for out in constrained_results])
        exceeded_per_instance= (vehicle_counts > opts.fleet_size).float().mean().item() #bools need to be float before mean
        constrained_cost= -constrained_rewards.mean().item()
        unconstrained_cost= -unconstrained_rewards.mean().item()
        masking_penalty =  (constrained_cost - unconstrained_cost) / unconstrained_cost
        inference_time = time.time() - start

        print(
            f"FLEET_SIZE_ONLY_{problem.upper()} | "
            f"Budget: {opts.fleet_size} | "
            f"Exceed rate: {exceeded_per_instance:.2f}% | "
            f"Unconstrained cost: {unconstrained_cost:.3f} | "
            f"Constrained cost: {constrained_cost:.3f} | "
            f"Masking penalty: {masking_penalty:.2f}% | "
            f"Inference time: {inference_time:.3f} s"
        )



        dataset_name = f"MULTI_FLEET_{opts.problem.upper()}"
        print(f"{dataset_name} | Cost: {-out['max_aug_reward'].mean().item():.3f} | Inference time: {inference_time:.3f} s")
        #print(f"{dataset_name} | Cost: {-out['max_aug_reward'].mean().item():.3f} | Gap: {out['gap_to_bks'].mean().item():.3f}% | Inference time: {inference_time:.3f} s")
        results[dataset_name] = {"cost": -out["max_aug_reward"].mean().item(),"inference_time": inference_time,}
        if opts.save_results:
            # Save results with checkpoint name under results/main/
            checkpoint_name = opts.checkpoint.split("/")[-1].split(".")[0]
            savedir = f"results/main/{opts.size}/"
            os.makedirs(savedir, exist_ok=True)
            pickle.dump(results, open(savedir + checkpoint_name + ".pkl", "wb"))

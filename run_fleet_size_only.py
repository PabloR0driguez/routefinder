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

FAILURE_REASONS = {
    1: "no_feasible_action",  5: "incomplete_trajectory",}

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
            out = policy(td, env, phase="test", num_starts=n_start, return_actions=True, select_best=False,)

            # Unbatchify reward to [batch_size, num_augment, num_starts].
            reward = unbatchify(out["reward"], (num_augment, n_start))
            details = getattr(env, "evaluation_result", None)
            if details is not None:
                reasons = unbatchify(details["reason"], (num_augment, n_start))
                steps = unbatchify(details["steps"], (num_augment, n_start))
                candidate_failed = unbatchify(details["failed"], (num_augment, n_start))
            else:
                candidate_failed = torch.zeros_like(reward, dtype=torch.bool)
                reasons = torch.zeros_like(reward, dtype=torch.long)
                steps = torch.zeros_like(reasons)
            out.update({
                "candidate_failed": candidate_failed,
                "candidate_failure_reason": reasons,
                "candidate_steps": steps,
            })

            #failed candidates cannot be chosen
            #copy of reward where every failed candidate gets -inf
            # successful candidates keep their score
            reward = reward.masked_fill(candidate_failed, -float("inf"))

            #best start within each augmentation
            max_reward, max_idxs = reward.max(dim=-1)
            out["max_reward"] = max_reward



            if out.get("actions", None) is not None:
                actions = unbatchify(out["actions"], (num_augment, n_start))
                out["best_multistart_actions"] = gather_by_index(
                    actions, max_idxs, dim=2
                )
                out["actions"] = actions

            # Get the best augmentation.
            max_aug_reward, max_idxs = max_reward.max(dim=1)

            out.update({
                "max_aug_reward": max_aug_reward,
                "has_solution": torch.isfinite(max_aug_reward),
            })

            if costs_bks is not None:
                gap_to_bks = (
                    100
                    * (-max_aug_reward - torch.abs(costs_bks))
                    / torch.abs(costs_bks)
                )
                out["gap_to_bks"] = gap_to_bks

            if out.get("actions", None) is not None:
                out["best_aug_actions"] = gather_by_index(
                    out["best_multistart_actions"],
                    max_idxs,
                )

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
        weights_only=False, load_baseline=False,
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
            variant_preset=problem, #variant_preset=opts.problem if opts.problem != "all" else "all",
            fleet_size=opts.fleet_size,
            fleet_mode="homogeneous",
        )
        env_constrained = MTVRPEnv(generator=generator, check_solution=False, enforce_fleet_limit=True,     handle_dead_ends=True,)
        env_unconstrained = MTVRPEnv(generator=generator, check_solution=False, enforce_fleet_limit=False,     handle_dead_ends=True,)

        # random seed
        torch.manual_seed(123)
        td_all = generator(batch_size=[opts.num_instances])
        dataloader = get_dataloader(td_all, batch_size=opts.batch_size)
        start = time.time()
        #res = []
        unconstrained_results, constrained_results, vehicle_counts = [], [], []


        for batch in dataloader:
            source_batch = batch.clone()

            # Keep an untouched initial state for the correctness check
            td_check_con = env_constrained.reset(source_batch.clone()).to(device)

            td_con = td_check_con.clone()
            o = test(policy,  td_con,  env_constrained, num_augment=8,  num_starts=None, device=device,)
            #Validate
            solved = o["has_solution"]
            if solved.any().item():
                env_constrained.check_solution_validity(td_check_con[solved], o["best_aug_actions"][solved],)
            constrained_results.append(o)

            # PENDING UNCONSTRAINED
            unconstrained_batch, num_customers = source_batch.clone(), source_batch["locs"].shape[-2] - 1
            for key in ("vehicle_capacities", "vehicle_speeds"):
                unconstrained_batch.set(key, source_batch[key][:, :1].expand(-1, num_customers).clone())
            unconstrained_batch.set("num_vehicles", torch.full_like(source_batch["num_vehicles"], num_customers))

            td_uncon = env_unconstrained.reset(unconstrained_batch).to(device)
            o_uncon = test(policy, td_uncon, env_unconstrained, num_augment=8, num_starts=None, device=device,)
            unconstrained_results.append(o_uncon)

            vehicles_used = count_vehicles_used(o_uncon["best_aug_actions"])
            vehicle_counts.append(vehicles_used)
            ####################

        out = {}
        vehicle_counts = torch.cat(vehicle_counts)
        unconstrained_rewards = torch.cat([out["max_aug_reward"] for out in unconstrained_results])
        constrained_rewards = torch.cat([out["max_aug_reward"] for out in constrained_results])

        unconstrained_solved = torch.cat([out["has_solution"] for out in unconstrained_results])
        constrained_solved = torch.cat([out["has_solution"] for out in constrained_results]) 
        num_instances = constrained_solved.numel()
        constrained_successes = int(constrained_solved.sum().item())
        unconstrained_successes = int(unconstrained_solved.sum().item())

        exceeded_per_instance= (vehicle_counts[unconstrained_solved] > opts.fleet_size).float().mean().item()  if unconstrained_successes else None#bools need to be float before mean
        constrained_cost= -constrained_rewards[constrained_solved].mean().item()  if constrained_successes else None
        unconstrained_cost= -unconstrained_rewards[unconstrained_solved].mean().item()   if unconstrained_successes else None
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
            f"FLEET_SIZE_ONLY_{problem.upper()} | "
            f"Budget: {opts.fleet_size} | "
            f"Exceed rate: {f'{exceeded_per_instance:.2%}' if exceeded_per_instance is not None else 'n/a'} | "
            f"Unconstrained cost: {f'{unconstrained_cost:.3f}' if unconstrained_cost is not None else 'n/a'} | "
            f"Constrained cost: {f'{constrained_cost:.3f}' if constrained_cost is not None else 'n/a'} | "
            f"Failures: constrained {num_instances - constrained_successes}, unconstrained {num_instances - unconstrained_successes} | "
            f"Masking penalty: {f'{masking_penalty:.2%}' if masking_penalty is not None else 'n/a'} | "
            f"Evaluation time: {inference_time:.3f} s"
        )

        dataset_name = f"MULTI_FLEET_{problem.upper()}"
        results[dataset_name] = {
            "cost": constrained_cost,
            "unconstrained_cost": unconstrained_cost,
            "exceedance_rate": exceeded_per_instance,
            "masking_penalty": masking_penalty,
            "constrained_failures": num_instances - constrained_successes,
            "unconstrained_failures": num_instances - unconstrained_successes,
            "evaluation_time": inference_time,
        }

        if opts.save_results:
            # Save results with checkpoint name under results/main/
            checkpoint_name = opts.checkpoint.split("/")[-1].split(".")[0]
            savedir = f"results/main/{opts.size}/"
            os.makedirs(savedir, exist_ok=True)
            pickle.dump(results, open(savedir + checkpoint_name + ".pkl", "wb"))
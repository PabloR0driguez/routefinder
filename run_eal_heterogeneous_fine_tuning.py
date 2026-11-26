# JUST BEGAN PENDING TO CHECK THIS WHOLE SCRIPT

"""Fine-tune current-vehicle EAL columns on continuous-fleet CVRP."""

import argparse
from pathlib import Path
import csv
import json
import time
import torch

# Reuse the evaluator's TorchRL compatibility setup and checkpoint loader.
from run_eal_heterogeneous import device, load_from_checkpoint_no_setup

from rl4co.data.utils import load_npz_to_tensordict
from rl4co.utils.ops import unbatchify
from routefinder.data.utils import get_dataloader
from routefinder.envs.mtvrp import MTVRPEnv, MTVRPGenerator
from routefinder.models.env_embeddings.mtvrp.context import (
    MTVRPContextEmbeddingCurrentVehicle, MTVRPContextEmbeddingFleet,
)
from routefinder.models.finetuning.eal import efficient_adapter_layers
from routefinder.models.model import RouteFinderBase
from run_fleet_size_only import test

"""Fine-tune current-vehicle or learned fleet context through EAL on continuous CVRP."""

import argparse
import csv
import json
import time
from pathlib import Path

import torch

# Reuse the evaluator's TorchRL compatibility setup and checkpoint loader.
from run_eal_heterogeneous import device, load_from_checkpoint_no_setup

from rl4co.data.utils import load_npz_to_tensordict
from rl4co.utils.ops import unbatchify
from routefinder.data.utils import get_dataloader
from routefinder.envs.mtvrp import MTVRPEnv, MTVRPGenerator
from routefinder.models.env_embeddings.mtvrp.context import (
    MTVRPContextEmbeddingCurrentVehicle,
    MTVRPContextEmbeddingFleet,
)
from routefinder.models.finetuning.eal import efficient_adapter_layers
from routefinder.models.model import RouteFinderBase
from run_fleet_size_only import test

VARIANTS = ["cvrp", "ovrp", "vrpb", "vrpl", "ovrpb", "ovrpl", "vrpbl", "ovrpbl"]

def create_run_folder(root):
    root.mkdir(parents=True, exist_ok=True)
    run_number = 1
    while True:
        folder = root / f"run_{run_number:03d}"
        try:
            folder.mkdir()
            return folder
        except FileExistsError:
            run_number += 1


def save_run(folder, context, num_new_features, step, config, metrics):
    weights = {
        "completed_steps": step,
        "new_eal_columns": (
            context.project_context.weight[:, -num_new_features:]
            .detach().cpu().clone()
        ),
    } #we select only the new EAL columns
    #with detach().cpu().clone() we detach the grad, move to cpu and make copy indepentdent

    if hasattr(context, "fleet_encoder"):
        weights["fleet_encoder_state_dict"] = {
            name: value.detach().cpu().clone()
            for name, value in context.fleet_encoder.state_dict().items()
        }

    temporary_weights = folder / "adaptation.tmp"
    torch.save(weights, temporary_weights)
    temporary_weights.replace(folder / "adaptation.pt")

    (folder / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    ) #save as json

    fields = [
        "stage", "problem", "step", "loss", "mean_unserved", "instances", "completed",
        "paired_instances", "paired_cost", "cost_change_pct", "elapsed_seconds",
    ]

    temporary_metrics = folder / "metrics.tmp"
    with temporary_metrics.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metrics)

    temporary_metrics.replace(folder / "metrics.csv")


def limit_fleet(instances, fleet_size):
    #enforcing the fleet limit out of the fleet pool, fleet pool simulates unconstrained
    for key in ("vehicle_capacities", "vehicle_speeds"):
        instances[key] = instances[key][:, :fleet_size].clone()
    instances["num_vehicles"] = torch.full_like(instances["num_vehicles"], fleet_size)


def validation_rewards(policy, env, instances, batch_size):
    rewards = []
    # Use the same evaluation seed and decoding settings before and after training.
    torch.manual_seed(124)
    for batch in get_dataloader(instances, batch_size=batch_size):
        initial_state = env.reset(batch.clone()).to(device)
        out = test(
            policy, initial_state.clone(), env,
            num_augment=8, num_starts=None, device=device,
        )
        solved = out["has_solution"]
        if solved.any().item():
            env.check_solution_validity(
                initial_state[solved], out["best_aug_actions"][solved]
            )
        rewards.append(out["max_aug_reward"].cpu())
    return torch.cat(rewards)


def print_comparison(before_rewards, after_rewards):
    before_solved = torch.isfinite(before_rewards)
    after_solved = torch.isfinite(after_rewards)

    print(f"Completed before: {before_solved.sum().item()}/{len(before_rewards)}")
    print(f"Completed after:  {after_solved.sum().item()}/{len(after_rewards)}")

    common = before_solved & after_solved
    before_cost, after_cost, cost_change_pct= None, None, None
    if common.any().item():
        before_cost = -before_rewards[common].mean().item()
        after_cost = -after_rewards[common].mean().item()
        print(
            f"Paired cost ({common.sum().item()} instances): "
            f"{before_cost:.4f} -> {after_cost:.4f}"
        )
        if before_cost != 0:
            cost_change_pct = 100 * (after_cost - before_cost) / before_cost
            print(f"Cost change: {cost_change_pct:.2f}%")
    else:
        print("No instances completed in both evaluations.")
    return {
        "paired_instances": int(common.sum().item()),
        "before_cost": before_cost,
        "after_cost": after_cost,
        "cost_change_pct": cost_change_pct,
    }




def main(args):
    # This pilot uses the CVRP cost bound in the failure penalty below.
    run_started = time.perf_counter()
    # CVRP filename only to ensure the expected naming structure 
    # and use it to make the matching filenames for the other variants
    dataset_path = Path(args.datasets)
    if dataset_path.name.split("_")[0].lower() != "cvrp":
        raise ValueError("CVRP dataset needed for naming")


    problem = Path(args.datasets).name.split("_")[0].lower()

    if problem != "cvrp":
        raise ValueError("Use a continuous-fleet CVRP validation dataset for this pilot.")
    if min(args.steps, args.num_instances, args.batch_size) < 1:
        raise ValueError("steps, num_instances and batch_size must be positive.")
    variants = VARIANTS if args.all_variants else ["cvrp"]
    train_batch_size = 16
    #both modes keep 16 instances per update
    #be it 16 CVRP or 2 from each of 8 variants
    instances_per_variant = train_batch_size // len(variants)
    validation_files = {"cvrp": dataset_path}
    if args.all_variants:
        suffix = dataset_path.name.split("_", 1)[1]
        validation_files = {
            problem: dataset_path.with_name(f"{problem}_{suffix}")
            for problem in variants
        }

    td_val = {}
    for problem in variants:
        print(f"Loading validation dataset: {problem.upper()}...", flush=True)
        instances = load_npz_to_tensordict(str(validation_files[problem]))[:args.num_instances].clone()
        if len(instances) == 0:
            raise ValueError(f"The {problem.upper()} validation dataset contains no instances.")
        td_val[problem] = instances
    #print("Loading validation dataset...", flush=True)
    #td_val = load_npz_to_tensordict(args.datasets)[:args.num_instances].clone()
    #if len(td_val) == 0:
    #    raise ValueError("The validation dataset contains no instances.")
    num_customers = td_val["cvrp"]["locs"].shape[-2] - 1 #cvrp is only working as a reference, take second to last dim, number of nodes -1 (depo)
    #pool_size = td_val["vehicle_capacities"].shape[-1]
    pool_size = td_val["cvrp"]["vehicle_capacities"].shape[-1]
    if not 1 <= args.fleet_size <= pool_size:
        raise ValueError(f"fleet_size must be between 1 and {pool_size}.")
    for instances in td_val.values():
        limit_fleet(instances, args.fleet_size)

    generators = {
        problem: MTVRPGenerator(
            num_loc=num_customers,
            variant_preset=problem,
            fleet_size=pool_size,
            fleet_mode="continuous",
        )
        for problem in variants
    }



    #generator = MTVRPGenerator(
    #    num_loc=num_customers,
    #    variant_preset="cvrp",
    #    fleet_size=pool_size,
    #    fleet_mode="continuous",
    #)
    env = MTVRPEnv(
        generator= generators["cvrp"],
        check_solution=False,
        enforce_fleet_limit=True,
        handle_dead_ends=True,
    )

    if args.fleet_context:
        torch.manual_seed(123)
    print("Loading pretrained model...", flush=True)
    model = load_from_checkpoint_no_setup(RouteFinderBase, args.checkpoint).to(device)
    
    num_new_features = (
        model.policy.decoder.context_embedding.embed_dim
        if args.fleet_context else 2
    )
    context_cls = (
        MTVRPContextEmbeddingFleet
        if args.fleet_context else MTVRPContextEmbeddingCurrentVehicle
        )


    num_new_features = (
        model.policy.decoder.context_embedding.embed_dim
        if args.fleet_context else 2
    )
    
    model = efficient_adapter_layers(
        model,
        # Keep the pretrained encoder embedding; only extend the decoder context.
        init_embedding_cls=None,
        init_embedding_num_new_feats=0,
        context_embedding_cls=context_cls,
        context_embedding_num_new_feats=num_new_features,
        adapter_only=True,
        new_features_only=True,
        train_fleet_encoder=args.fleet_context,
    )

    # Keep normalization statistics fixed. Training below explicitly samples routes.
    policy = model.policy.eval()
    context_weight = policy.decoder.context_embedding.project_context.weight
    weights_before = context_weight.detach().clone()

    #next three 4 lines are not relevant, only for debugging
    if args.fleet_context:
        fleet_encoder = policy.decoder.context_embedding.fleet_encoder
        fleet_weights_before = [
            p.detach().clone() for p in fleet_encoder.parameters()
        ]
    ###
    # With the old gradients masked, a fresh optimizer and zero weight decay
    # preserve the old context columns.
    #optimizer = torch.optim.Adam([context_weight], lr=args.lr, weight_decay=0.0)

    trainable_parameters = [ p for p in policy.parameters() if p.requires_grad]

    optimizer = torch.optim.Adam(trainable_parameters, lr=args.lr, weight_decay=0.0,)
    # I AM HERE; WERE WE ADD INFO TO SAVE THE WEIGHTS OF THE FINE TUNING
    mode = "fleet_context" if args.fleet_context else "current_vehicle"
    run_root = Path(__file__).resolve().parent / args.output_dir / mode
    run_folder = create_run_folder(run_root)

    print("Saving run to:", run_folder, flush=True)

    config = {
        **vars(args),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "datasets": str(Path(args.datasets).resolve()),
        "run_folder": str(run_folder),
        "problem": "all" if args.all_variants else "cvrp",
        "variants": variants,
        "validation_datasets": {problem: str(path.resolve()) for problem, path in validation_files.items()},
        "fleet_mode": "continuous",
        "num_customers": num_customers,
        "stored_fleet_size": pool_size,
        "validation_instances": sum(len(instances) for instances in td_val.values()),
        "validation_instances_per_variant": {problem: len(instances) for problem, instances in td_val.items()},
        "context_class": context_cls.__name__,
        "embed_dim": policy.decoder.context_embedding.embed_dim,
        "num_new_features": num_new_features,
        "initialization_seed": 123 if args.fleet_context else None,
        "training_seed": 123,
        "validation_seed": 124,
        "train_batch_size": 16,
        "train_instances_per_variant_per_update": instances_per_variant,
        "training_instances_per_variant": args.steps * instances_per_variant,
        "train_num_starts": 8,
        "train_decode_type": "multistart_sampling",
        "validation_num_augment": 8,
        "validation_augment_fn": "dihedral8",
        "validation_num_starts": "env.get_num_starts(td)",
        "validation_decode_type": policy.test_decode_type,
        "penalty_per_unserved_customer": 3.0 * num_customers,
        "gradient_clip_norm": 1.0,
        "weight_decay": 0.0,
        "save_every_updates": 25,
        "device": str(device),
        "torch_version": str(torch.__version__),
    }

    if args.fleet_context:
        config["fleet_encoder_num_heads"] = (
            fleet_encoder.veh_encoder_self_attention.num_heads
        )

    (run_folder / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )


    ### --- ###
    #print("Validation before training...", flush=True)
    #before_rewards = validation_rewards(policy, env, td_val, args.batch_size)
    before_rewards = {}
    metrics = []
    for problem in variants:
        print(f"Val before train: {problem.upper()}...", flush=True,)
        rewards = validation_rewards(policy, env, td_val[problem], args.batch_size)
        before_rewards[problem] = rewards

        metrics.append({
            "stage": "validation_before",
            "problem": problem.upper(),
            "step": 0,
            "instances": len(rewards),
            "completed": int(torch.isfinite(rewards).sum().item()),
            "elapsed_seconds": round(time.perf_counter() - run_started, 3),
        })   
    #metrics = [{... }]

    ### --- ###


    # Generate fresh training instances independently of the saved validation set.
    torch.manual_seed(123)
    train_num_starts = 8
    # Complete unit-square CVRP solutions with speeds >= 1 cost less than 3N.
    # Use that bound for the penalty per unserved customer, as in the dual pilot.
    penalty = 3.0 * num_customers

    for step in range(1, args.steps + 1):
        #train_batch = generator([16])
        #go for each variant
        train_batch = torch.cat([generators[problem]([instances_per_variant]) for problem in variants], dim=0)
        limit_fleet(train_batch, args.fleet_size)
        td_train = env.reset(train_batch).to(device)

        out = policy(
            td_train, env,
            phase="train",
            decode_type="multistart_sampling",
            num_starts=train_num_starts,
            select_best=False,
            return_actions=False,
        )

        details = env.evaluation_result
        route_cost = details["route_cost"]
        violation = details["unserved_customers"]
        reward = -(route_cost + penalty * violation)
        reward = unbatchify(reward, train_num_starts)
        log_likelihood = unbatchify(out["log_likelihood"], train_num_starts)

        # Compare each sampled route with the mean reward for its instance.
        advantage = (reward - reward.mean(dim=-1, keepdim=True)).detach()
        loss = -(advantage * log_likelihood).mean()
        if not torch.isfinite(loss).item():
            raise RuntimeError("Training loss is not finite.")

        optimizer.zero_grad()
        loss.backward()
        # Limit unusually large updates from sampled trajectories.
        torch.nn.utils.clip_grad_norm_( trainable_parameters, 1.0, error_if_nonfinite=True)
        #torch.nn.utils.clip_grad_norm_([context_weight], 1.0, error_if_nonfinite=True)
        optimizer.step()


        if step == 1 or step % 25 == 0 or step == args.steps:
            metrics.append({
                "stage": "train",
                "step": step,
                "loss": loss.item(),
                "mean_unserved": violation.float().mean().item(),
                "instances": 16,
                "elapsed_seconds": round(time.perf_counter() - run_started, 3),
            })

            # Save at the existing progress intervals to retain training progress.
            save_run(
                run_folder,
                policy.decoder.context_embedding,
                num_new_features,
                step,
                config,
                metrics,
            )

            print(
                f"Update {step}/{args.steps} | Loss: {loss.item():.4f} | "
                f"Mean unserved: {violation.float().mean().item():.2f}",
                flush=True,
            )


    print("Old columns unchanged:", torch.equal(weights_before[:, :-num_new_features], context_weight[:, :-num_new_features]))
    print("New columns changed:", not torch.equal(weights_before[:, -num_new_features:], context_weight[:, -num_new_features:]))
    # next 10 lines are only for debugging, have no effect
    if args.fleet_context:
        fleet_encoder_changed = False
        for before, after in zip(
            fleet_weights_before, fleet_encoder.parameters()
        ):
            if not torch.equal(before, after):
                fleet_encoder_changed = True
                break
        print("Fleet encoder changed:", fleet_encoder_changed)
    
    
    ### ---- ###
    comparisons = []

    for index, problem in enumerate(variants):
        print(f"Val after train: {problem.upper()}...", flush=True, )

        after_rewards = validation_rewards(policy, env, td_val[problem], args.batch_size)
        comparison = print_comparison(before_rewards[problem], after_rewards)
        metrics[index].update({
            "paired_instances": comparison["paired_instances"],
            "paired_cost": comparison["before_cost"],
        })
        metrics.append({
            "stage": "validation_after",
            "problem": problem.upper(),
            "step": args.steps,
            "instances": len(after_rewards),
            "completed": int(torch.isfinite(after_rewards).sum().item()),
            "paired_instances": comparison["paired_instances"],
            "paired_cost": comparison["after_cost"],
            "cost_change_pct": comparison["cost_change_pct"],
            "elapsed_seconds": round(time.perf_counter() - run_started, 3),
        })
        comparisons.append({
            "problem": problem.upper(),
            "instances": len(after_rewards),
            "completed_before": int(torch.isfinite(before_rewards[problem]).sum().item()),
            "completed_after": int(torch.isfinite(after_rewards).sum().item()),
            **comparison,
        })

    with (run_folder / "comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=list(comparisons[0])
        )
        writer.writeheader()
        writer.writerows(comparisons)


    ### ---- ###

    save_run(
        run_folder,
        policy.decoder.context_embedding,
        num_new_features,
        args.steps,
        config,
        metrics,
    )

    print("Saved weights, configuration and metrics to:", run_folder, flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/100/rf-transformer.ckpt")
    parser.add_argument("--datasets", required=True, help="Continuous-fleet CVRP validation NPZ")
    parser.add_argument("--fleet_size", type=int, required=True)
    parser.add_argument("--num_instances", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4, help="Validation batch size")
    # Use a fixed update budget to keep this CVRP pilot bounded.
    parser.add_argument("--steps", type=int, default=500)
    # Start with the original RouteFinder EAL learning rate.
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--fleet_context", action="store_true")
    parser.add_argument("--output_dir", default="results_hypothesis/heterogeneous", help="Where to save run folders",)
    parser.add_argument("--all_variants",action="store_true", help="train and val the eight used variants, not only CVRP",
)
    main(parser.parse_args())

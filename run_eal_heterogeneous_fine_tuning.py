# JUST BEGAN PENDING TO CHECK THIS WHOLE SCRIPT

"""Fine-tune current-vehicle EAL columns on continuous-fleet CVRP."""

import argparse
from pathlib import Path

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

    # Compare route cost only on instances completed in both evaluations.
    common = before_solved & after_solved
    if common.any().item():
        before_cost = -before_rewards[common].mean().item()
        after_cost = -after_rewards[common].mean().item()
        print(f"Paired cost ({common.sum().item()} instances): {before_cost:.4f} -> {after_cost:.4f}")
    else:
        print("No instances completed in both evaluations.")


def main(args):
    # This pilot uses the CVRP cost bound in the failure penalty below.
    problem = Path(args.datasets).name.split("_")[0].lower()
    if problem != "cvrp":
        raise ValueError("Use a continuous-fleet CVRP validation dataset for this pilot.")
    if min(args.steps, args.num_instances, args.batch_size) < 1:
        raise ValueError("steps, num_instances and batch_size must be positive.")

    print("Loading validation dataset...", flush=True)
    td_val = load_npz_to_tensordict(args.datasets)[:args.num_instances].clone()
    if len(td_val) == 0:
        raise ValueError("The validation dataset contains no instances.")
    num_customers = td_val["locs"].shape[-2] - 1
    pool_size = td_val["vehicle_capacities"].shape[-1]
    if not 1 <= args.fleet_size <= pool_size:
        raise ValueError(f"fleet_size must be between 1 and {pool_size}.")
    limit_fleet(td_val, args.fleet_size)

    generator = MTVRPGenerator(
        num_loc=num_customers,
        variant_preset="cvrp",
        fleet_size=pool_size,
        fleet_mode="continuous",
    )
    env = MTVRPEnv(
        generator=generator,
        check_solution=False,
        enforce_fleet_limit=True,
        handle_dead_ends=True,
    )


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


    print("Validation before training...", flush=True)
    before_rewards = validation_rewards(policy, env, td_val, args.batch_size)

    # Generate fresh training instances independently of the saved validation set.
    torch.manual_seed(123)
    train_num_starts = 8
    # Complete unit-square CVRP solutions with speeds >= 1 cost less than 3N.
    # Use that bound for the penalty per unserved customer, as in the dual pilot.
    penalty = 3.0 * num_customers

    for step in range(1, args.steps + 1):
        train_batch = generator([16])
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
    
    
    
    print("Validation after training...", flush=True)
    after_rewards = validation_rewards(policy, env, td_val, args.batch_size)
    print_comparison(before_rewards, after_rewards)


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
    main(parser.parse_args())

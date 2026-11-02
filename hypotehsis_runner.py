
from pathlib import Path
import torch
import run_eal_dual
from routefinder.envs.mtvrp import MTVRPEnv, MTVRPGenerator
from run_fleet_size_only import test, count_vehicles_used

# Settings
checkpoint = "checkpoints/100/rf-transformer.ckpt"
num_loc = 20
num_instances = 100
budget = 10
seed = 321
num_augment = 8

variants = [
    "cvrp", "ovrp", "vrpb", "vrpl", "ovrpb", "ovrpl", "vrpbl", "vrptw",
    "ovrpbl", "ovrptw", "vrpbtw", "vrpltw", "ovrpbtw", "ovrpltw", "vrpbltw", "ovrpbltw",
]


scenarios = [ ("fleet_only", "homogeneous", budget),   ("minimal", "continuous", budget),  ("dual", "two_type", budget), ]






print(f"Fleet budget: {budget}. , locs: {num_loc}")

def gen_instances(variant, variant_index):

    #Generate customers first once and pair them with the the three settings
    torch.manual_seed(seed + variant_index)
    generator = MTVRPGenerator(num_loc=num_loc, variant_preset=variant, fleet_size=num_loc, fleet_mode="continuous")
    data = generator([num_instances])
    cases = {}
    return generator, data


model = run_eal_dual.load_from_checkpoint_no_setup(run_eal_dual.RouteFinderBase, checkpoint).cpu()
model.requires_grad_(False)
policy = model.policy.eval()


print("fleet exceedance use completed solutions; failure % uses all instances.")

for variant_index, variant in enumerate(variants):
    generator, data = gen_instances(variant, variant_index)

    env = MTVRPEnv(generator=generator, check_solution=False, enforce_fleet_limit=False, handle_dead_ends=True)
    td_check = env.reset(data.clone())
    if variant_index == 0:
        print("Fleet mode:", env.generator.fleet_mode)
        print("First five capacities:", td_check["vehicle_capacities"][0, :5].tolist())
        print("First five speeds:", td_check["vehicle_speeds"][0, :5].tolist())
        print(f"Variant| Complete| Incomp| Veh_exced| Over_budget%_")


    out = test(policy, td_check.clone(), env, num_augment=num_augment, num_starts=num_loc, device="cpu")

    solved = out["has_solution"]
    actions = out["best_aug_actions"]
    successes = int(solved.sum().item())
    failures = num_instances - successes
    failure_percent = 100 * (num_instances - successes) / num_instances

    if successes:
        env.check_solution_validity(td_check[solved], actions[solved])
        #cost = -out["max_aug_reward"][solved].mean().item()
        vehicles = count_vehicles_used(actions[solved])
        
        violations = int((vehicles > budget).sum().item())
        violation_count = str(violations)
        violation_percent = f"{100 * violations / successes:.2f}%"
    else:
        violation_count = "n/a"
        violation_percent = "n/a"

    print(f"{variant.upper():<10} | {successes} | {failures} | {violation_count} | {violation_percent}", flush=True)


print("Evaluation complete.")




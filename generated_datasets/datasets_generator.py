import torch

from routefinder.envs.mtvrp import MTVRPGenerator

from pathlib import Path
from rl4co.data.utils import save_tensordict_to_npz

def generate_mtvrp_instances(
    num_instances=10,
    num_loc=20,
    fleet_size=5,
    variant_preset="all",
    continuous_bool=False,
    dual_bool=False,
    homogeneous_bool=False,
    seed=None,
    split="test"
):
    split_seeds = {"train": 123, "val": 124, "test": 125}
    if seed is None:
        seed = split_seeds[split]

    generated_instances = {}
    output_root = Path(__file__).resolve().parent / "output"

    filename = (
        f"{variant_preset}"
        f"_n{num_loc}"
        f"_instances{num_instances}"
        f"_fleet{fleet_size}"
        f"_seed{seed}.npz"
    )

    if continuous_bool:
        torch.manual_seed(seed)

        generator = MTVRPGenerator(
            num_loc=num_loc,
            variant_preset=variant_preset,
            fleet_size=fleet_size,
            fleet_mode="continuous",
        )

        td = generator(torch.Size([num_instances]))
        generated_instances["continuous"] = td

        continuous_folder = output_root / "continuous"/ split
        continuous_folder.mkdir(parents=True, exist_ok=True)

        continuous_path = continuous_folder / filename
        save_tensordict_to_npz(td, str(continuous_path))

        print(f"Saved continuous instances to: {continuous_path}")

    if dual_bool:
        torch.manual_seed(seed)

        generator = MTVRPGenerator(
            num_loc=num_loc,
            variant_preset=variant_preset,
            fleet_size=fleet_size,
            fleet_mode="two_type",
        )

        td = generator(torch.Size([num_instances]))
        generated_instances["dual"] = td

        dual_folder = output_root / "dual"/ split
        dual_folder.mkdir(parents=True, exist_ok=True)

        dual_path = dual_folder / filename
        save_tensordict_to_npz(td, str(dual_path))

        print(f"Saved dual instances to: {dual_path}")

    if homogeneous_bool:
        torch.manual_seed(seed)

        generator = MTVRPGenerator(
            num_loc=num_loc,
            variant_preset=variant_preset,
            fleet_size=fleet_size,
            fleet_mode="homogeneous",
        )

        td = generator(torch.Size([num_instances]))
        generated_instances["homogeneous"] = td

        homogeneous_folder = output_root / "homogeneous"/ split
        homogeneous_folder.mkdir(parents=True, exist_ok=True)

        homogeneous_path = homogeneous_folder / filename
        save_tensordict_to_npz(td, str(homogeneous_path))

        print(f"Saved homogeneous instances to: {homogeneous_path}")

    if not generated_instances:
        raise ValueError(
            "At least one generation mode must be True."
        )

    return generated_instances




if __name__ == "__main__":
    # Save evaluation data; training data is generated on the fly.
    split_sizes = {"val": 1000, "test": 1000}
    num_loc = 100
    fleet_size = num_loc
    variants = ["cvrp", "ovrp", "vrpb", "vrpl", "ovrpb", "ovrpl", "vrpbl", "ovrpbl"]
    total = len(split_sizes) * len(variants)
    current = 0
    for split, num_instances in split_sizes.items():
        for variant_preset in variants:
            current += 1
            print(
                f"\n[{current}/{total}] Generating {split} | "
                f"{variant_preset.upper()} | {num_instances} instances "
                f"per fleet mode",
                flush=True,)            
            generate_mtvrp_instances(
                num_instances=num_instances,
                num_loc=num_loc,
                fleet_size=fleet_size,
                variant_preset=variant_preset,
                continuous_bool=True,
                dual_bool=True,
                homogeneous_bool=True,
                split=split,
            )
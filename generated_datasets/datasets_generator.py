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
    seed=123,
):
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

        continuous_folder = output_root / "continuous"
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

        dual_folder = output_root / "dual"
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

        homogeneous_folder = output_root / "homogeneous"
        homogeneous_folder.mkdir(parents=True, exist_ok=True)

        homogeneous_path = homogeneous_folder / filename
        save_tensordict_to_npz(td, str(homogeneous_path))

        print(f"Saved homogeneous instances to: {homogeneous_path}")

    if not generated_instances:
        raise ValueError(
            "At least one generation mode must be True."
        )

    return generated_instances
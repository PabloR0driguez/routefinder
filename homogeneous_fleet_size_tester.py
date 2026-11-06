# purpose, identify where the fleet budget makes a difference
import subprocess
import sys

dataset = "generated_datasets/output/homogeneous/val/cvrp_n100_instances1000_fleet100_seed124.npz"

num_instances = 100
results = []

for fleet_size in [10, 11, 12, 13]:
    print(f"\nFleet budget: {fleet_size}", flush=True)
    total = 0
    summary = None    
    output = subprocess.run(
        [
            sys.executable, "-u", "run_fleet_size_only.py",
            "--checkpoint", "checkpoints/100/rf-transformer.ckpt",
            "--datasets", dataset,
            "--problem", "cvrp",
            "--fleet_size", str(fleet_size),
            "--num_instances", "100",
            "--batch_size", "4",
            "--device", "cuda",
            "--no-save-results",
        ],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout
    summary = None
    for line in output.splitlines():
        if line.startswith("FLEET_SIZE_ONLY_CVRP"):
            summary = line
            break
    _, _, rate, failures, *_ = summary.split(" | ")
    solved = num_instances - int(failures.split()[-1])
    violations = round(float(rate.split()[-1].rstrip("%")) * solved / 100) if solved else "n/a"
    results.append(f"{summary} | Violations: {violations}")

print("\nSummary")
print("\n".join(results))

#run with uv run python -u .\homogeneous_fleet_size_tester
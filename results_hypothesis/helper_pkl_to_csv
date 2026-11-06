import csv
import pickle
from pathlib import Path

#folder = Path(__file__).parent
folder = Path(__file__).resolve().parent / "homogeneous"
rows = []

#for path in sorted(folder.glob("*.pkl")):
for path in sorted(folder.rglob("*.pkl")):
    with path.open("rb") as file:
        results = pickle.load(file)

    for name, result in results.items():
        unrestricted_solved = 1000 - result["unconstrained_failures"]
        rows.append({
            "problem": name.removeprefix("MULTI_FLEET_"),
            "fleet_budget": 11,
            "instances": 1000,
            "violations": round(result["exceedance_rate"] * unrestricted_solved),
            "constrained_solved": 1000 - result["constrained_failures"],
            "masking_penalty_pct": 100 * result["masking_penalty"],
            "source_file": str(path.relative_to(folder)),
        })

with (folder / "fleet_size_results.csv").open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


#uv run python .\results_hypothesis\helper_pkl_to_csv.py
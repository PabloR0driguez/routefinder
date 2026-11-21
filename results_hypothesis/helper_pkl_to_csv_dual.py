import csv
import pickle
from pathlib import Path

#folder = Path(__file__).resolve().parent / "dual"
folder = (
    Path(__file__).resolve().parent
    / "dual"
    / "dual_budget29_n1000_ngk7nlru-20261107T085305Z-1-001"
    / "dual_budget29_n1000_ngk7nlru"
)

rows = []

for path in sorted(folder.rglob("*.pkl")):
    with path.open("rb") as file:
        results = pickle.load(file)

    for name, result in results.items():
        rows.append({
            "problem": name.removeprefix("DUAL_FLEET_"),
            "fleet_budget": result["fleet_budget"],
            "instances": result["instances"],
            "violations": result["violations"],
            "exceedance_rate_pct": (
                100 * result["exceedance_rate"]
                if result["exceedance_rate"] is not None else None
            ),
            "unconstrained_solved": result["unconstrained_solved"],
            "constrained_solved": result["constrained_solved"],
            "masking_penalty_pct": result["masking_penalty_pct"],
            "source_file": str(path.relative_to(folder)),
        })

if not rows:
    raise SystemExit(f"No pickle results found in {folder}")

output_file = folder / "dual_fleet_size_results.csv"

with output_file.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved {len(rows)} rows to {output_file}")


#uv run python .\results_hypothesis\helper_pkl_to_csv_dual.py
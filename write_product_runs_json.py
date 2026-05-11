import os
import json
import sys

KEEP = 4

if len(sys.argv) != 3:
    raise SystemExit("Usage: python write_product_runs_json.py <model> <product>")

model = sys.argv[1]
product = sys.argv[2]

runs_dir = os.path.join("site", "runs", model, product)
os.makedirs(runs_dir, exist_ok=True)

runs = sorted(
    [
        d for d in os.listdir(runs_dir)
        if os.path.isdir(os.path.join(runs_dir, d))
    ],
    reverse=True
)[:KEEP]

out = {
    "model": model,
    "product": product,
    "runs": runs
}

json_path = os.path.join(runs_dir, "runs.json")

with open(json_path, "w") as f:
    json.dump(out, f, indent=2)

print(f"Wrote {json_path}")
print(runs)

import os
import shutil


KEEP_COUNTS = {
    ("hrrr", "refl_uh"): 4,
    ("hrrr", "hail_swath"): 4,
    ("rrfs", "refl_uh"): 4,
    ("rrfs", "hail_swath"): 4,
    ("refs_mem01", "refl_uh"): 1,
}


def cleanup_runs(model, product, keep):

    base_dir = os.path.abspath(
        os.path.join("site", "runs", model, product)
    )

    print(f"\nCleaning: {base_dir}")

    if not os.path.exists(base_dir):
        print("Directory does not exist.")
        return

    runs = sorted(
        [
            d for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d))
        ],
        reverse=True
    )

    print(f"Found {len(runs)} runs")

    old_runs = runs[keep:]

    for run in old_runs:
        path = os.path.join(base_dir, run)

        print(f"Removing old run: {path}")

        shutil.rmtree(path, ignore_errors=True)


# ============================================================
# REMOVE OLD LEGACY RUN FOLDERS
# ============================================================

legacy_base = os.path.join("site", "runs")

if os.path.exists(legacy_base):

    for name in os.listdir(legacy_base):

        path = os.path.join(legacy_base, name)

        if (
            os.path.isdir(path)
            and name.startswith("20")
            and "_z" in name
        ):

            print(f"Removing legacy run folder: {path}")

            shutil.rmtree(path, ignore_errors=True)


# ============================================================
# CLEAN MODERN RUN STRUCTURE
# ============================================================

for (model, product), keep in KEEP_COUNTS.items():
    cleanup_runs(model, product, keep)

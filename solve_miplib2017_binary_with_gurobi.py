#!/usr/bin/env python3
"""Solve the transformed MIPLIB 2017 binary instances to optimality with Gurobi.

The best known objective values obtained here serve as the reference for
computing the primal gap and primal integral metrics of Section 6. For each
instance the script stores the solving statistics in a gzip-pickled
dictionary (--output-pkl) and the best solution as a .sol file
(--solution-dir); existing results can be skipped or resumed via
--skip-existing / --resume-unsolved.
"""

import argparse
import glob
import gzip
import os
import pickle
import re
import sys

try:
    import gurobipy as gp
except ImportError as e:
    raise ImportError("gurobipy is required to run this script. Install it in your environment.") from e


def parse_args():
    parser = argparse.ArgumentParser(description="Solve transformed miplib2017 MPS models with Gurobi and save objectives/solutions.")
    parser.add_argument(
        "--instance-dir",
        type=str,
        default="./data/generated_instances/miplib2017_binary/-small/transformedmodel/test/",
        help="Directory containing transformed MPS instances."
    )
    parser.add_argument(
        "--output-pkl",
        type=str,
        default="./result/miplib2017/miplib2017_transformed_opt_data.pkl.gz",
        help="Output gzip-pickled objective dictionary."
    )
    parser.add_argument(
        "--solution-dir",
        type=str,
        default="./result/miplib2017/miplib2017_transformed_solutions/",
        help="Directory to save solution files (.sol)."
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=3600.0,
        help="Time limit per model in seconds."
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="Number of Gurobi threads. Use 0 for automatic."
    )
    parser.add_argument(
        "--root-solution-dir",
        type=str,
        default="./data/generated_instances/miplib2017_binary/-small/rootsol/test/",
        help="Directory containing root solution .sol files to use as Gurobi MIP starts."
    )
    parser.add_argument(
        "--instance",
        type=str,
        default=None,
        help="Optional instance base name to solve only one instance, e.g. miplib2017_binary-41_transformed"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-solve instances even if output pickle already contains results."
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip instances that already have results in the output pickle."
    )
    parser.add_argument(
        "--resume-unsolved",
        action="store_true",
        help="Only solve instances that are not marked as solved to optimality in the existing output pickle."
    )
    return parser.parse_args()


def find_instance_files(instance_dir, selected_instance=None):
    pattern = os.path.join(instance_dir, "*_transformed.mps")
    files = glob.glob(pattern)
    if selected_instance is not None:
        if not selected_instance.endswith("_transformed"):
            selected_instance = selected_instance + "_transformed"
        files = [f for f in files if os.path.splitext(os.path.basename(f))[0] == selected_instance]
    if not files:
        raise FileNotFoundError(f"No transformed MPS files found in {instance_dir} matching {pattern}")

    def sort_key(path):
        base = os.path.splitext(os.path.basename(path))[0]
        match = re.search(r"-(\d+)_transformed$", base)
        return int(match.group(1)) if match else base

    return sorted(files, key=sort_key)


def load_existing_results(output_pkl):
    if not os.path.exists(output_pkl):
        return {}
    with gzip.open(output_pkl, "rb") as f:
        return pickle.load(f)


def save_results(output_pkl, results):
    os.makedirs(os.path.dirname(output_pkl), exist_ok=True)
    with gzip.open(output_pkl, "wb") as f:
        pickle.dump(results, f)


def load_gurobi_mip_start(model, sol_path):
    if not os.path.exists(sol_path):
        return False

    var_map = {var.VarName: var for var in model.getVars()}
    start_vars = []
    start_vals = []

    with open(sol_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.lower().startswith("objective value"):
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            var_name = parts[0]
            try:
                value = float(parts[1])
            except ValueError:
                continue

            if var_name in var_map:
                start_vars.append(var_map[var_name])
                start_vals.append(value)

    if not start_vars:
        return False

    model.setAttr("Start", start_vars, start_vals)
    return True


def solve_model_with_gurobi(mps_path, time_limit, threads, solution_dir, root_solution_dir=None):
    instance_name = os.path.splitext(os.path.basename(mps_path))[0]
    print(f"Solving {instance_name}...", flush=True)
    model = gp.read(mps_path)
    model.setParam("TimeLimit", time_limit)
    model.setParam("OutputFlag", 0)
    if threads > 0:
        model.setParam("Threads", threads)

    mip_start_path = None
    previous_sol_path = os.path.join(solution_dir, f"{instance_name}.sol")
    if os.path.exists(previous_sol_path):
        mip_start_path = previous_sol_path

    if mip_start_path is None and root_solution_dir is not None:
        root_sol_name = f"rootsol-{instance_name}.sol"
        root_sol_path = os.path.join(root_solution_dir, root_sol_name)
        if os.path.exists(root_sol_path):
            mip_start_path = root_sol_path

    if mip_start_path is not None:
        if not load_gurobi_mip_start(model, mip_start_path):
            print(f"  no compatible MIP start in {mip_start_path}", flush=True)
    else:
        print("  no MIP start available", flush=True)

    model.optimize()

    status = int(model.Status)
    status_name = gp.GRB.StatusName(status) if hasattr(gp.GRB, "StatusName") else str(status)
    runtime = model.Runtime
    sol_count = int(model.SolCount) if hasattr(model, "SolCount") else 0
    best_obj = float(model.ObjVal) if sol_count > 0 else None
    best_bound = float(model.ObjBound) if model.ObjBound not in (gp.GRB.INFINITY, float("inf"), None) else None
    mip_gap = None
    if sol_count > 0 and best_obj is not None and best_bound is not None and best_bound != 0:
        mip_gap = abs(best_obj - best_bound) / abs(best_bound)
    solved_to_optimal = (status == gp.GRB.OPTIMAL)

    if sol_count > 0:
        os.makedirs(solution_dir, exist_ok=True)
        sol_name = f"{instance_name}.sol"
        sol_path = os.path.join(solution_dir, sol_name)
        model.write(sol_path)
    else:
        sol_path = None

    result = {
        "instance": instance_name,
        "status": status_name,
        "status_code": status,
        "solve_time": runtime,
        "solved_to_optimal": solved_to_optimal,
        "best_obj": best_obj,
        "best_bound": best_bound,
        "mip_gap": mip_gap,
        "sol_count": sol_count,
        "solution_path": sol_path,
        "mps_path": os.path.abspath(mps_path),
    }
    print(
        f"  summary: status={status_name}, time={runtime:.2f}s, solved_to_optimal={solved_to_optimal}, "
        f"best_obj={best_obj}, best_bound={best_bound}, mip_gap={mip_gap}, sols={sol_count}",
        flush=True,
    )
    return result


def main():
    args = parse_args()
    instance_files = find_instance_files(args.instance_dir, selected_instance=args.instance)
    results = load_existing_results(args.output_pkl)
    os.makedirs(args.solution_dir, exist_ok=True)

    selected_files = []
    for mps_path in instance_files:
        instance_name = os.path.splitext(os.path.basename(mps_path))[0]
        if args.skip_existing and instance_name in results:
            print(f"Skipping {instance_name} because it is already present in {args.output_pkl}")
            continue
        if args.resume_unsolved:
            existing_entry = results.get(instance_name)
            if existing_entry is not None and existing_entry.get("solved_to_optimal") is True:
                print(f"Skipping {instance_name} because it was already solved to optimality")
                continue
        selected_files.append(mps_path)

    for mps_path in selected_files:
        instance_name = os.path.splitext(os.path.basename(mps_path))[0]
        try:
            results[instance_name] = solve_model_with_gurobi(
                mps_path,
                args.time_limit,
                args.threads,
                args.solution_dir,
                root_solution_dir=args.root_solution_dir,
            )
            save_results(args.output_pkl, results)
        except gp.GurobiError as e:
            print(f"GurobiError solving {instance_name}: {e}")
        except Exception as e:
            print(f"Error solving {instance_name}: {e}")

    print(f"Saved {len(results)} entries to {args.output_pkl}")


if __name__ == "__main__":
    main()

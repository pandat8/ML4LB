"""Print the seed-averaged Section 6 results (Tables 11 and 12).

Variant of compute_evaluation_results_scip.py that averages results across
multiple random seeds. It assumes that each seed run produced comparison
files in the usual result directories; the script calls the same comparison
helper with a per-seed suffix and then aggregates the returned statistics
(per-instance averages, solved counts, speedups and gap reductions, also at
intermediate cutoff times).

Usage example:
    python compute_evaluation_results_scip_seeds_averaged.py \
        --seeds 2021 2022 2023 2024 2025 --dataset_id 5 --t_total 3600 --mean geometric --enable_gpu
"""

import ecole
import numpy as np
import pyscipopt
import argparse
from ml4lb.execute_heuristics import ExecuteHeuristic
from ml4lb.utilities import instancetypes, instancesizes, lbconstraint_mode_for
import torch
import random
import pathlib
import os
import csv
from scipy.stats import gmean

parser = argparse.ArgumentParser()
parser.add_argument('--seeds', type=int, nargs='+',
                    default=[2021, 2022, 2023, 2024, 2025],
                    help='List of random seeds to average over')
parser.add_argument('--mean', type=str, default='geometric',
                    help="averaging mode for the metrics: 'arithmetic' or 'geometric'")
parser.add_argument('--dataset_id', type=int, default=4,
                    help='dataset to aggregate, index into ml4lb.utilities.instancetypes '
                         "(4: 'miplib_39binary', 5: 'miplib2017_binary')")
parser.add_argument('--t_total', type=int, default=3600,
                    help='total time limit (s) of the evaluation runs to aggregate')
parser.add_argument('--t_node', type=int, default=2,
                    help='node time limit (s) of the evaluation runs to aggregate')
parser.add_argument('--enable_gpu', action='store_true', help='Enable CUDA GPU acceleration')
parser.add_argument('--cutoffs', type=int, nargs='+', default=[3600, 1800, 1200, 600, 60],
                    help='List of cutoff times (seconds) for evaluation')
args = parser.parse_args()

cutoff_times = args.cutoffs
print('Cutoff times:', cutoff_times)

# Device tag used in the result directory names of the LB runs.
enable_gpu = args.enable_gpu
if enable_gpu and torch.cuda.is_available():
    device_str = 'cuda'
else:
    device_str = 'cpu'

mean_option = args.mean
print(str(mean_option))

total_time_limit = args.t_total
node_time_limit = args.t_node
print('total time limit:', total_time_limit)
print('node time limit:', node_time_limit)

# Metrics are additionally reported at these intermediate cutoff times
# (only cutoffs within the total time limit are kept, without duplicates).
eval_cutoff_times = [total_time_limit]
for cutoff in cutoff_times:
    cutoff = int(cutoff)
    if cutoff > 0 and cutoff <= total_time_limit and cutoff not in eval_cutoff_times:
        eval_cutoff_times.append(cutoff)

# Select the dataset and the LB constraint mode used for it in the paper.
instance_type = instancetypes[args.dataset_id]
lbconstraint_mode = lbconstraint_mode_for(instance_type)

# Section 6 aggregates the small-size runs started from the root solution.
for incumbent_mode in ['rootsol']:
    for instance_size in [instancesizes[0]]:

        print(instance_type + instance_size)
        print(incumbent_mode)
        print(lbconstraint_mode)

        # Output directory for the generated plots and comparison tables.
        plots_directory = './result/plots/'
        pathlib.Path(plots_directory).mkdir(parents=True, exist_ok=True)

        # Base of the result directories written by the evaluation scripts
        # (the per-seed sub-directories are appended below).
        evaluation_directory = './result/generated_instances/' + instance_type + '/' + instance_size + '/' + incumbent_mode + '/' + 'scip/'
        evaluation_directory = evaluation_directory + 'heuristic_mode/'

        # Step 1: run the single-seed comparison for every seed and collect
        # the returned metrics and per-instance records.
        seed_results = []
        for seed in args.seeds:
            print(f"Processing seed {seed}")
            # set random seeds for reproducibility
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            # Result directories of this seed: the SCIP baseline (rd1) and
            # the scip-lb-regression-rl runs with freq 0, 1 and 100 (rd2-rd4);
            # they must match the paths written by the evaluation scripts.
            rd1 = evaluation_directory + 'lb-from-' + incumbent_mode + '-t_total' + str(
                total_time_limit) + 's' + instance_size + '_scip_baseline' + '-' + 'cpu' + '/seed' + str(seed) + '/'
            rd2 = evaluation_directory + 'lb-from-' + incumbent_mode + '-t_total' + str(
                total_time_limit) + 's' + '-t_node' + str(node_time_limit) + 's' + instance_size + '_lb_k0_regression_rl_beforenode_freq_0' + '-' + device_str + '/seed' + str(seed) + '/'
            rd3 = evaluation_directory + 'lb-from-' + incumbent_mode + '-t_total' + str(
                total_time_limit) + 's' + '-t_node' + str(
                node_time_limit) + 's' + instance_size + '_lb_k0_regression_rl_beforenode_freq_1' + '-' + device_str + '/seed' + str(seed) + '/'
            rd4 = evaluation_directory + 'lb-from-' + incumbent_mode + '-t_total' + str(
                total_time_limit) + 's' + '-t_node' + str(
                node_time_limit) + 's' + instance_size + '_lb_k0_regression_rl_beforenode_freq_100' + '-' + device_str + '/seed' + str(seed) + '/'

            # Input directories: test instances and their stored incumbents.
            source_directory = './data/generated_instances/' + instance_type + '/' + instance_size + '/'
            instance_directory = source_directory + 'transformedmodel' + '/' + 'test/'
            solution_directory = source_directory + incumbent_mode + '/' + 'test/'

            run_localbranch = ExecuteHeuristic(instance_type, instance_directory, solution_directory, rd1, seed=seed)

            # Run the comparison of this seed; a per-seed CSV is written
            # (csv_suffix) and the metrics are returned for averaging.
            res = run_localbranch.primal_integral_scip_comparison(
                seed_mcts=seed,
                instance_type=instance_type,
                instance_size=instance_size,
                incumbent_mode=incumbent_mode,
                total_time_limit=total_time_limit,
                node_time_limit=node_time_limit,
                mean_option=mean_option,
                result_directory_1=rd1,
                result_directory_2=rd2,
                result_directory_3=rd3,
                result_directory_4=rd4,
                csv_suffix=f"_seed{seed}_v2_202607",
                cutoff_times=eval_cutoff_times,
            )
            seed_results.append(res)

        # Step 2: average the aggregate metrics over the seeds.
        if seed_results:
            # determine numeric keys to average
            numeric_keys = [k for k in seed_results[0].keys() if k not in ('per_instance_results',
                                                                              'seed_mcts',
                                                                              'instance_type',
                                                                              'instance_size',
                                                                              'incumbent_mode')]
            avg_metrics = {}
            for k in numeric_keys:
                values = np.array([r[k] for r in seed_results])
                avg_metrics[k] = values.mean()

            print("\n=== Seed-averaged metrics ===")
            for k, v in avg_metrics.items():
                print(f"{k}: {v}")

            # Metric keys of the three scip-lb-regression-rl variants used in
            # the summary table below.
            mapping = {
                'freq0': ('primal_int_freq0_ave', 'primal_gap_final_freq0_ave'),
                'freq1': ('primal_int_freq1_ave', 'primal_gap_final_freq1_ave'),
                'freq100': ('primal_int_freq100_ave', 'primal_gap_final_freq100_ave'),
            }

            # Step 3: average the per-instance records over the seeds
            # (solve times, final gaps and solved flags per instance).
            per_instance_map = {}
            for r in seed_results:
                for rec in r['per_instance_results']:
                    inst = rec['instance']
                    per_instance_map.setdefault(inst, []).append(rec)
            averaged_instances = []
            for inst, recs in per_instance_map.items():
                agg = {'instance': inst}
                keys = [kk for kk in recs[0].keys() if kk != 'instance']
                for kk in keys:
                    vals = [rr[kk] for rr in recs]
                    # For boolean 'solved' flags, calculate the proportion of times it was true.
                    # This is equivalent to the mean, but more explicit.
                    # For the summary, we will treat >= 0.5 as "solved on average".
                    if isinstance(vals[0], (bool, np.bool_)):
                        agg[kk] = np.mean(vals)
                    else:
                        try:
                            agg[kk] = np.mean(vals)
                        except Exception:
                            agg[kk] = vals[0]
                averaged_instances.append(agg)

            # Step 4: write the seed-averaged per-instance table (the input
            # of print_appendix_table.py).
            out_csv = f"./result/plots/scip_comparison_details_{instance_type}_{instance_size}_{incumbent_mode}_seeds_averaged_v2_202607.csv"
            os.makedirs(os.path.dirname(out_csv), exist_ok=True)
            with open(out_csv, 'w', newline='') as csvf:
                writer = csv.DictWriter(csvf, fieldnames=averaged_instances[0].keys())
                writer.writeheader()
                for rec in averaged_instances:
                    writer.writerow(rec)
            print(f"Averaged per-instance table written to {out_csv}")

            # Step 5: print the comparison summaries from the averaged records.
            def _summarize(records, suffix="", label="averaged"):
                """Print speedup/gap-reduction summaries of each method vs the baseline.

                Two groupings are reported: the intersection method (instances
                solved/unsolved by both the baseline and the method) and the
                fixed-baseline method (instances grouped only by the baseline
                status).
                """
                for method in ['freq0', 'freq1', 'freq100']:
                    # --- OLD METHOD (Intersection - solved on at least ONE seed by both methods) ---
                    solved = [r for r in records if r[f'baseline_solved{suffix}'] > 0 and r[f'{method}_solved{suffix}'] > 0]
                    unsolved = [r for r in records if r[f'baseline_solved{suffix}'] == 0 and r[f'{method}_solved{suffix}'] == 0]
                    print(f"\n--- {method} vs baseline ({label}) [OLD METHOD - Intersection] ---")
                    print(f"affected_solved: {len(solved)}")
                    if solved:
                        base = np.array([r[f'baseline_solve_time{suffix}'] for r in solved])
                        meth = np.array([r[f'{method}_solve_time{suffix}'] for r in solved])
                        positive = meth > 0
                        if not positive.all():
                            n_bad = (np.logical_not(positive)).sum()
                            print(f"  warning: {n_bad} {method}_solve_time{suffix} <= 0 entries skipped when computing speedups")
                        valid_base = base[positive]
                        valid_meth = meth[positive]
                        if valid_meth.size > 0:
                            speedups = valid_base / valid_meth
                            abs_red = valid_base - valid_meth
                            n_faster = (abs_red > 0).sum()
                            n_slower = (abs_red < 0).sum()
                            print(f"  speedup mean/med/geo: {speedups.mean():.2f}/{np.median(speedups):.2f}/{gmean(speedups):.2f}")
                            print(f"  time reduction mean/med: {abs_red.mean():.2f}/{np.median(abs_red):.2f} s"
                                  f"  (faster: {n_faster}, slower: {n_slower})")
                        else:
                            print("  no valid solve times to compute speedups")
                    print(f"affected_unsolved: {len(unsolved)}")
                    if unsolved:
                        gap_red = np.array([r[f'baseline_final_gap{suffix}'] - r[f'{method}_final_gap{suffix}'] for r in unsolved])
                        rel_red = gap_red / np.array([r[f'baseline_final_gap{suffix}'] if r[f'baseline_final_gap{suffix}'] > 0 else np.nan for r in unsolved]) * 100
                        print(f"  gap reduction mean/med: {gap_red.mean():.2f}/{np.median(gap_red):.2f} pp")
                        print(f"  rel gap red mean/med: {rel_red.mean():.2f}/{np.nanmedian(rel_red):.2f} %")

                    # --- FAIR METHOD (Fixed Baseline Subsets) ---
                    b_solved = [r for r in records if r[f'baseline_solved{suffix}'] > 0]
                    b_unsolved = [r for r in records if r[f'baseline_solved{suffix}'] == 0]
                    print(f"\n--- {method} vs baseline ({label}) [FAIR METHOD - Fixed Baseline] ---")

                    print(f"b_solved (baseline solved): {len(b_solved)}")
                    if b_solved:
                        base = np.array([r[f'baseline_solve_time{suffix}'] for r in b_solved])
                        meth = np.array([r[f'{method}_solve_time{suffix}'] for r in b_solved])
                        positive = meth > 0
                        regressed_count = sum(1 for r in b_solved if r[f'{method}_solved{suffix}'] == 0)
                        print(f"  regressed to timeout: {regressed_count}")

                        if not positive.all():
                            n_bad = (np.logical_not(positive)).sum()
                            print(f"  warning: {n_bad} {method}_solve_time{suffix} <= 0 entries skipped when computing speedups")

                        valid_base = base[positive]
                        valid_meth = meth[positive]

                        if valid_meth.size > 0:
                            speedups = valid_base / valid_meth
                            abs_red = valid_base - valid_meth
                            n_faster = (abs_red > 0).sum()
                            n_slower = (abs_red < 0).sum()
                            print(f"  speedup mean/med/geo: {speedups.mean():.2f}/{np.median(speedups):.2f}/{gmean(speedups):.2f}")
                            print(f"  time reduction mean/med: {abs_red.mean():.2f}/{np.median(abs_red):.2f} s"
                                  f"  (faster: {n_faster}, slower: {n_slower})")
                        else:
                            print("  no valid solve times to compute speedups")

                    print(f"b_unsolved (baseline unsolved): {len(b_unsolved)}")
                    if b_unsolved:
                        newly_solved_count = sum(1 for r in b_unsolved if r[f'{method}_solved{suffix}'] > 0)
                        print(f"  newly solved: {newly_solved_count}")

                        gap_red = np.array([r[f'baseline_final_gap{suffix}'] - r[f'{method}_final_gap{suffix}'] for r in b_unsolved])
                        rel_red = gap_red / np.array([r[f'baseline_final_gap{suffix}'] if r[f'baseline_final_gap{suffix}'] > 0 else np.nan for r in b_unsolved]) * 100
                        print(f"  gap reduction mean/med: {gap_red.mean():.2f}/{np.median(gap_red):.2f} pp")
                        print(f"  rel gap red mean/med: {rel_red.mean():.2f}/{np.nanmedian(rel_red):.2f} %")

            def _count_solved_on_any_seed(records, method, suffix=""):
                """Count instances solved by the method on at least one seed.

                The averaged 'solved' flag is the proportion of seeds on which
                the instance was solved, so > 0 means solved at least once.
                """
                key = f'{method}_solved{suffix}'
                return sum(1 for r in records if r.get(key, 0) > 0)

            def _append_suffix_to_ave_key(metric_key, suffix):
                """Insert the cutoff suffix into a '*_ave' metric key name."""
                if not suffix:
                    return metric_key
                if metric_key.endswith('_ave'):
                    return metric_key.replace('_ave', f'{suffix}_ave')
                return f'{metric_key}{suffix}'

            # Print the summary table and comparisons for the full time limit
            # and every intermediate cutoff time (Tables 11 and 12).
            for cutoff in eval_cutoff_times:
                suffix = "" if cutoff == total_time_limit else f"_{cutoff}"
                label = f"averaged {cutoff}s"
                print(f"\n--- Proposed algorithms (5‑seed averages, {cutoff}s) ---")
                print("method   | count_optimal | primal_int         | primal_gap_final")
                print("---------+---------------+--------------------+-------------------")
                base_count = _count_solved_on_any_seed(averaged_instances, 'baseline', suffix)
                base_pi_key = _append_suffix_to_ave_key('primal_int_scip_baseline_ave', suffix)
                base_gap_key = _append_suffix_to_ave_key('primal_gap_final_scip_baseline_ave', suffix)
                base_pi = avg_metrics.get(base_pi_key, float('nan'))
                base_gap = avg_metrics.get(base_gap_key, float('nan'))
                print(f"{'baseline':7} | {base_count:13.1f} | {base_pi:18.6f} | {base_gap:17.6f}")
                for method, (pi_key, gap_key) in mapping.items():
                    count = _count_solved_on_any_seed(averaged_instances, method, suffix)
                    pi = avg_metrics.get(_append_suffix_to_ave_key(pi_key, suffix), float('nan'))
                    gap = avg_metrics.get(_append_suffix_to_ave_key(gap_key, suffix), float('nan'))
                    print(f"{method:7} | {count:13.1f} | {pi:18.6f} | {gap:17.6f}")

                _summarize(averaged_instances, suffix=suffix, label=label)

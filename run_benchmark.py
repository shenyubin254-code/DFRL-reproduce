import argparse
import csv
import hashlib
import json
import re
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def summarize_run(results, objective, similarity, seed):
    def values(column):
        return [float(row[column]) for row in results]

    def average(column):
        return statistics.mean(values(column))

    qed_before = values("initial_qed")
    qed_after = values("optimized_qed")
    drd2_before = values("initial_drd2")
    drd2_after = values("optimized_drd2")
    logp_before = values("initial_logp")
    logp_after = values("optimized_logp")

    metrics = {
        "mean_qed": statistics.mean(qed_after),
        "mean_logp": statistics.mean(logp_after),
        "mean_drd2": statistics.mean(drd2_after),
        "mean_similarity": average("similarity"),
        "mean_qed_improvement": statistics.mean(
            after - before for before, after in zip(qed_before, qed_after)
        ),
        "mean_logp_improvement": statistics.mean(
            after - before for before, after in zip(logp_before, logp_after)
        ),
        "mean_drd2_improvement": statistics.mean(
            after - before for before, after in zip(drd2_before, drd2_after)
        ),
    }

    if objective == "qed":
        metrics["qed_ge_0_9_percent"] = 100 * sum(value >= 0.9 for value in qed_after) / len(results)
        metrics["qed_delta_ge_0_1_percent"] = 100 * sum(
            after - before >= 0.1 for before, after in zip(qed_before, qed_after)
        ) / len(results)
    elif objective == "plogp":
        metrics["logp_improvement"] = metrics["mean_logp_improvement"]
    elif objective == "drd2":
        metrics["drd2_ge_0_5_percent"] = 100 * sum(value >= 0.5 for value in drd2_after) / len(results)
        metrics["drd2_delta_ge_0_2_percent"] = 100 * sum(
            after - before >= 0.2 for before, after in zip(drd2_before, drd2_after)
        ) / len(results)
    elif objective == "qed&drd2":
        metrics["mo_success_percent"] = 100 * sum(
            qed > 0.6 and drd2 > 0.5 for qed, drd2 in zip(qed_after, drd2_after)
        ) / len(results)

    return {
        "objective": objective,
        "min_similarity": similarity,
        "seed": seed,
        "num_molecules": len(results),
        **metrics,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run repeated DFRL-Mol optimization experiments and aggregate paper-style metrics."
    )
    parser.add_argument(
        "--objectives",
        nargs="+",
        default=["qed"],
        choices=["qed", "plogp", "drd2", "qed&drd2"],
        help="Optimization tasks to run. Use 'qed&drd2' for the paper's multi-objective task.",
    )
    parser.add_argument("--similarities", nargs="+", type=float, default=[0.6])
    parser.add_argument("--runs", type=int, default=1, help="Independent seeds; default 1 is a staged reproduction, not repeated-run statistics.")
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--epochs_test", type=int, default=20)
    parser.add_argument("--swarm_num", type=int, default=20)
    parser.add_argument("--spa_max_length", type=int, default=50)
    parser.add_argument("--spa_model_path", default="models/spa")
    parser.add_argument("--dcr_model_path", default="models/dcr")
    parser.add_argument("--data_fname_test", default="")
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--resume", action="store_true", help="Resume the same explicit output_dir; skip verified complete runs.")
    parser.add_argument("--use_one_mask", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow_individuals_repeat", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.resume and not args.output_dir:
        raise ValueError("--resume requires --output_dir")
    if args.runs < 1:
        raise ValueError("--runs must be at least 1")
    if not args.similarities or any(not 0 <= value <= 1 for value in args.similarities):
        raise ValueError("--similarities must contain values between 0 and 1")

    output_dir = Path(args.output_dir) if args.output_dir else (
        Path("benchmark_results") /
        datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    if output_dir.exists() and not args.resume:
        raise FileExistsError(
            f"Benchmark directory already exists: {output_dir}"
        )
    manifest_file = output_dir / "benchmark_config.json"
    parameters = {k: v for k, v in vars(args).items() if k not in ('resume', 'output_dir')}
    # Hash inputs once per invocation, also protecting the completed-run skip path.
    inputs = list(Path('.').glob('*.py')) + list(Path('oracle').glob('*.py'))
    inputs += ([Path(args.data_fname_test)] if args.data_fname_test
               else list(Path('data/standard_test_data').glob('*.csv')))
    for model_path in (args.spa_model_path, args.dcr_model_path):
        inputs += [p for p in Path(model_path).rglob('*') if p.is_file()]
    import os
    inputs.append(Path(os.environ.get('DRD2_MODEL_PATH', 'models/drd2/clf_py36.pkl')))
    manifest = {'parameters': parameters,
                'inputs': {str(p.resolve()): file_hash(p) for p in sorted(set(inputs))},
                'scope': 'single-seed staged reproduction' if args.runs == 1 else 'repeated-seed reproduction'}
    if args.resume:
        if not manifest_file.exists() or json.loads(manifest_file.read_text(encoding='utf-8')) != manifest:
            raise ValueError("Resume refused: missing manifest or changed experiment inputs. Legacy outputs remain untouched.")
    run_dir = output_dir / "runs"
    run_dir.mkdir(parents=True, exist_ok=args.resume)
    if not args.resume:
        manifest_file.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f"Output directory: {output_dir.resolve()} ({manifest['scope']})", flush=True)

    run_metrics = []
    molecule_results = []
    for objective in args.objectives:
        objective_name = re.sub(r"[^A-Za-z0-9_-]+", "_", objective)
        for similarity in args.similarities:
            for run_index in range(args.runs):
                seed = args.seed_start + run_index
                similarity_tag = f"{similarity:.2f}".replace(".", "_")
                result_file = run_dir / (
                    f"{objective_name}_sim_{similarity_tag}_seed_{seed}.csv"
                )
                command = [
                    sys.executable,
                    "-u",
                    "optimize.py",
                    "--epochs_test", str(args.epochs_test),
                    "--swarm_num", str(args.swarm_num),
                    "--min_similar", str(similarity),
                    "--spa_model_path", args.spa_model_path,
                    "--dcr_model_path", args.dcr_model_path,
                    "--opti_object", objective,
                    "--spa_max_length", str(args.spa_max_length),
                    "--seed", str(seed),
                    "--output_file", str(result_file),
                ]
                if args.data_fname_test:
                    command.extend(["--data_fname_test", args.data_fname_test])
                if args.use_one_mask:
                    command.append("--use_one_mask")
                if args.allow_individuals_repeat:
                    command.append("--allow_individuals_repeat")

                details_dir = result_file.parent / f"{result_file.stem}_details"
                complete_file = details_dir / "complete.json"
                completed = args.resume and complete_file.exists()
                if completed:
                    completed_info = json.loads(complete_file.read_text(encoding='utf-8'))
                    if file_hash(result_file) != completed_info['csv_sha256']:
                        raise ValueError(f"Completed CSV checksum mismatch: {result_file}")
                elif args.resume and details_dir.exists():
                    command.append("--resume")

                print(
                    f"Running objective={objective}, "
                    f"similarity={similarity}, seed={seed}",
                    flush=True
                )
                log_file = result_file.with_suffix(".log")
                with log_file.open("a" if args.resume else "x", encoding="utf-8") as log_stream:
                    if completed:
                        print(f"Verified complete; skipping {result_file}", flush=True)
                    else:
                        run_process(command, log_stream)

                with result_file.open(newline="") as result_stream:
                    results = list(csv.DictReader(result_stream))
                if not results:
                    raise ValueError(f"Empty result file: {result_file}")
                run_metrics.append(summarize_run(results, objective, similarity, seed))
                molecule_results.extend({
                    "objective": objective,
                    "min_similarity": similarity,
                    "seed": seed,
                    **row,
                } for row in results)
                write_csv(
                    output_dir / "per_run_metrics.csv",
                    run_metrics)
                write_csv(
                    output_dir / "summary.csv",
                    aggregate_metrics(run_metrics))
                write_csv(
                    output_dir / "all_molecule_results.csv",
                    molecule_results)

    print(f"Benchmark results saved to {output_dir}")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def run_process(command, log_stream):
    process = subprocess.Popen(command, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1)
    try:
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_stream.write(line)
            log_stream.flush()
        return_code = process.wait()
    except BaseException:
        process.terminate()
        process.wait()
        raise
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def aggregate_metrics(run_metrics):
    metric_columns = dict.fromkeys(
        key
        for row in run_metrics
        for key in row
        if key not in ("objective", "min_similarity", "seed", "num_molecules")
    )
    grouped_metrics = {}
    for row in run_metrics:
        for metric in metric_columns:
            if metric in row:
                key = (row["objective"], row["min_similarity"], metric)
                grouped_metrics.setdefault(key, []).append(row[metric])
    return [
        {
            "objective": objective,
            "min_similarity": similarity,
            "metric": metric,
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else "",
            "runs": len(values),
            "scope": "single_seed_no_variance_estimate" if len(values) == 1 else "repeated_seeds",
        }
        for (objective, similarity, metric), values in grouped_metrics.items()
    ]


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    fieldnames = list(dict.fromkeys(
        key
        for row in rows
        for key in row
    ))
    temp_path = path.with_name(path.name + ".tmp")
    with temp_path.open(
        "w", newline="", encoding="utf-8"
    ) as output_stream:
        writer = csv.DictWriter(
            output_stream,
            fieldnames=fieldnames
        )
        writer.writeheader()
        writer.writerows(rows)
    temp_path.replace(path)


if __name__ == "__main__":
    main()

import argparse
import glob
import json
import os.path
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path

from git import Repo

from huggingface_hub import HfApi

from optimum_benchmark import Benchmark
from optimum_benchmark_wrapper import main

# AI Features
from ai_feature_module import AIDrivenMetricSelection, AIBasedOptimization

PATH_TO_REPO = Path(__file__).parent.parent.resolve()

@contextmanager
def checkout_commit(repo: Repo, commit_id: str):
    """
    Context manager that checks out a given commit when entered, but gets back to the reference it was at on exit.
    Args:
        repo (`git.Repo`): A git repository (for instance the Transformers repo).
        commit_id (`str`): The commit reference to checkout inside the context manager.
    """
    current_head = repo.head.commit if repo.head.is_detached else repo.head.ref

    try:
        repo.git.checkout(commit_id)
        yield
    finally:
        repo.git.checkout(current_head)

def summarize(run_dir, metrics, expand_metrics=False):
    """Produce a summary for each optimum-benchmark launched job's output directory found in `run_dir`."""

    def process_report(report_dir, commit):
        """Process each benchmark report directory to extract summary."""
        if not os.path.isfile(os.path.join(report_dir, "benchmark.json")):
            return None

        benchmark = Benchmark.from_json(os.path.join(report_dir, "benchmark.json"))
        report = benchmark.report
        model = benchmark.config.backend["model"]
        benchmark_name = get_benchmark_name(report_dir, benchmark.config.name)
        metrics_values = extract_metrics(report, metrics, expand_metrics)
        return {
            "model": model,
            "commit": commit,
            "config": benchmark_name,
            "metrics": metrics_values,
        }

    def get_benchmark_name(report_dir, default_name):
        """Extract benchmark name from directory path."""
        benchmark_name = re.sub(f"backend.model={model},*", "", report_dir)
        benchmark_name = str(Path(benchmark_name).parts[-1])
        if benchmark_name.startswith("commit="):
            benchmark_name = default_name
        return benchmark_name

    def extract_metrics(report, metrics, expand_metrics):
        """Extract and format metrics from the benchmark report."""
        metrics_values = {}
        selected_metrics = AIDrivenMetricSelection(metrics, report.to_dict()).select_metrics()
        for metric in selected_metrics:
            value = traverse_metric(report.to_dict(), metric)
            if expand_metrics:
                update_metrics_values(metrics_values, metric, value)
            else:
                metrics_values[metric] = value
        return metrics_values

    def traverse_metric(data, metric):
        """Traverse nested dictionary to get the metric value."""
        keys = metric.split(".")
        for key in keys:
            if isinstance(data, dict) and key in data:
                data = data[key]
            else:
                return None
        return data

    def update_metrics_values(metrics_values, metric, value):
        """Update the metrics values dictionary with expanded metrics."""
        keys = metric.split(".")
        current = metrics_values
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value

    def save_summary(report_dir, summary):
        """Save the summary to a JSON file."""
        with open(os.path.join(report_dir, "summary.json"), "w") as fp:
            json.dump(summary, fp, indent=4)

    # Main processing
    reports = glob.glob(os.path.join(run_dir, "**/benchmark_report.json"), recursive=True)
    report_dirs = [str(Path(report).parent) for report in reports]

    summaries = []
    for report_dir in report_dirs:
        commit = re.search(r"/commit=([^/]+)", report_dir).groups()[0]
        summary = process_report(report_dir, commit)
        if summary:
            summaries.append(summary)
            save_summary(report_dir, summary)

    return summaries

def combine_summaries(summaries):
    """Combine a list of summary obtained from the function `summarize`."""

    combined = {}
    for summary in summaries:
        model = summary["model"]
        config = summary["config"]
        commit = summary["commit"]

        if model not in combined:
            combined[model] = {}

        if config not in combined[model]:
            combined[model][config] = {}

        if commit not in combined[model][config]:
            combined[model][config][commit] = {"metrics": summary["metrics"]}

    save_combined_summary(combined)
    return combined

def save_combined_summary(combined):
    """Save the combined summary to a JSON file."""
    combined_summary_path = os.path.join(exp_run_dir, "summary.json")
    with open(combined_summary_path, "w") as fp:
        json.dump(combined, fp, indent=4)
    print(json.dumps(combined, indent=4))

if __name__ == "__main__":

    def list_str(values):
        return values.split(",")

    parser = argparse.ArgumentParser()

    parser.add_argument("--config-dir", type=str, required=True, help="The path to the config directory.")
    parser.add_argument("--config-name", type=str, required=True, help="The config name.")

    # Arguments specific to this wrapper for our own customization
    parser.add_argument("--ensure_empty", type=bool, default=True, help="If to create a temporary directory.")
    parser.add_argument(
        "--commit",
        type=list_str,
        default="",
        help="Comma-separated list of branch names and/or commit SHA values on which the benchmark will run. If `diff` is specified, it will run on both the current head and the `main` branch.",
    )
    parser.add_argument("--metrics", type=str, help="The metrics to be included in the summary.")

    parser.add_argument("--repo_id", type=str, default=None, help="The repository to which the file will be uploaded.")
    parser.add_argument("--path_in_repo", type=str, default=None, help="Relative filepath in the repo.")
    parser.add_argument("--token", type=str, default=None, help="A valid user access token (string).")

    args, optimum_benchmark_args = parser.parse_known_args()

    repo = Repo(PATH_TO_REPO)

    # AI-Based Optimization for Benchmark Configuration
    AIBasedOptimization(args, optimum_benchmark_args)

    metrics = [
        "prefill.latency.mean",
        "prefill.throughput.value",
        "decode.latency.mean",
        "decode.throughput.value",
        "per_token.latency.mean",
        "per_token.throughput.value",
    ]
    if args.metrics is not None:
        metrics = args.metrics.split(",")

    # Get `backend.model` in a hacky way: We want to control the experiment flow manually.
    models = [""]
    for idx, arg in enumerate(optimum_benchmark_args):
        if arg.startswith("backend.model="):
            models = arg[len("backend.model="):]
            models = models.split(",")
            break
    optimum_benchmark_args = [arg for arg in optimum_benchmark_args if not arg.startswith("backend.model=")]

    # Get the commit(s)
    current_head = str(repo.head.commit) if repo.head.is_detached else str(repo.head.ref)
    commits = [x for x in args.commit if x != ""]
    if len(commits) == 0:
        commits = [current_head]
    elif len(commits) == 1 and commits[0] == "diff":
        # Compare to `main`
        commits = ["main", current_head]

    # Get the specified run directory
    run_dir_arg_idx, run_dir = -1, None
    sweep_dir_arg_idx, sweep_dir = -1, None
    for idx, arg in enumerate(optimum_benchmark_args):
        if arg.startswith("run_dir="):
            run_dir_arg_idx, run_dir = idx, arg.split("=")[1]
        elif arg.startswith("sweep_dir="):
            sweep_dir_arg_idx, sweep_dir = idx, arg.split("=")[1]

    with tempfile.TemporaryDirectory() as tmpdir:
        if args.ensure_empty:
            exp_run_dir = run_dir
            run_dir = tmpdir

        summaries = []
        for model in models:
            for commit_id in commits:
                with checkout_commit(repo, commit_id):
                    repo.git.clean("-xdf")
                    repo.git.submodule("update", "--init", "--recursive")

                    # Run the benchmark
                    if model:
                        cmd = [f"backend.model={model}"] + optimum_benchmark_args
                    else:
                        cmd = optimum_benchmark_args

                    main(["--config-dir", args.config_dir, "--config-name", args.config_name] + cmd)
                    if sweep_dir is not None:
                        summaries += summarize(sweep_dir, metrics)
                    else:
                        summaries += summarize(run_dir, metrics)

        # Finalize the results
        if args.ensure_empty:
            exp_run_dir = exp_run_dir or sweep_dir or tmpdir
            for idx, directory in [(run_dir_arg_idx, exp_run_dir), (sweep_dir_arg_idx, exp_run_dir)]:
                if idx >= 0:
                    optimum_benchmark_args[idx] = "=".join(optimum_benchmark_args[idx].split("=")[:-1] + [directory])

            os.makedirs(exp_run_dir, exist_ok=True)
            os.system(f"cp -a {Path(run_dir)}/* {exp_run_dir}/")

        combined_summary = combine_summaries(summaries)

        # Store the results on Hugging Face Hub if applicable
        if args.repo_id:
            api = HfApi()
            api.upload_file(
                path_or_fileobj=os.path.join(exp_run_dir, "summary.json"),
                repo_id=args.repo_id,
                path_in_repo=args.path_in_repo or "summary.json",
                token=args.token,
            )

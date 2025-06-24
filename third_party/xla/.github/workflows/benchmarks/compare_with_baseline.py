# Copyright 2025 The OpenXLA Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Compares benchmark results to predefined baselines and checks for regressions."""
import argparse
import json
import os
import sys
import yaml


def load_results_data(results_json_file):
  """Loads and parses the results JSON file."""
  try:
    with open(results_json_file, "r") as f:
      results_data = json.load(f)
  except json.JSONDecodeError as e:
    print(
        f"::error file={results_json_file}::Failed to parse results JSON: {e}"
    )
    sys.exit(1)
  except OSError as e:  # Catch other potential file reading errors
    print(f"::error file={results_json_file}::Error reading results JSON: {e}")
    sys.exit(1)
  return results_data


def load_baseline_data(baseline_yaml_file):
  """Loads and parses the baseline YAML file."""
  try:
    with open(baseline_yaml_file, "r") as f:
      baseline_data_full = yaml.safe_load(f)
  except yaml.YAMLError as e:
    print(
        f"::error file={baseline_yaml_file}::Failed to parse baseline YAML: {e}"
    )
    sys.exit(1)
  except OSError as e:  # Catch other potential file reading errors
    print(
        f"::error file={baseline_yaml_file}::Error reading baseline YAML: {e}"
    )
    sys.exit(1)
  return baseline_data_full


def validate_baseline_data(baseline_data_full, config_id, baseline_yaml_file):
  """Validates the loaded baseline data and extracts config-specific baselines."""
  if not baseline_data_full or not isinstance(baseline_data_full, dict):
    print(
        f"::warning file={baseline_yaml_file}::Baseline YAML is empty or"
        " not a dictionary. Skipping comparison."
    )
    sys.exit(0)

  if config_id not in baseline_data_full:
    print(
        f"::notice::No baseline found for config_id '{config_id}' in"
        f" {baseline_yaml_file}. Skipping comparison."
    )
    sys.exit(0)

  config_baselines = baseline_data_full[config_id]
  if not isinstance(config_baselines, dict):
    print(
        f"::warning file={baseline_yaml_file},title=Invalid Baseline"
        f" Structure::Baseline entry for '{config_id}' is not a"
        " dictionary. Skipping."
    )
    sys.exit(0)
  return config_baselines


def compare_metrics(
    results_data, config_baselines, results_json_file, config_id
):
  """Compares metrics from results to baselines and reports findings."""
  regressions_found = False
  summary_messages = ["\n--- Comparison Summary ---"]

  # --- IMPORTANT: Metric Extraction Logic ---
  # This assumes your results.json (produced by compute_xspace_stats_main or
  # fallback)
  # has a structure like:
  # {
  #   ...,
  #   "metrics": {
  #     "GPU_DEVICE_TIME": { "value_ms": 150.0, "unit": "ms", ... },
  #     "GPU_DEVICE_MEMCPY_TIME": { "value_ms": 1.2, "unit": "ms", ... },
  #     ...
  #   }
  # }
  # If your actual results.json structure is different, you MUST adapt this
  # section.
  actual_metrics_container = results_data.get("metrics")

  if not actual_metrics_container or not isinstance(
      actual_metrics_container, dict
  ):
    summary_messages.append(
        "::warning title=Missing Metrics in Results::'metrics' key not found"
        f" or not a dictionary in '{results_json_file}'. Cannot perform"
        " comparison."
    )
    # Depending on your policy, if metrics are always expected,
    # this could be sys.exit(1)
    # For now, it will skip comparisons and pass if no metrics are found.
    print("\n".join(summary_messages))
    sys.exit(0)  # Exit cleanly if no metrics to compare

  for metric_name, baseline_info in config_baselines.items():
    if not isinstance(baseline_info, dict) or not all(
        key in baseline_info for key in ["baseline_ms", "threshold"]
    ):
      summary_messages.append(
          f"::warning title=Malformed Baseline::Metric '{metric_name}' in"
          f" baseline for '{config_id}' is missing 'baseline_ms' or"
          " 'threshold', or is not structured as a dictionary. Skipping."
      )
      continue

    try:
      baseline_value_ms = float(baseline_info["baseline_ms"])
      threshold_percentage = float(baseline_info["threshold"])
    except ValueError:
      summary_messages.append(
          f"::warning title=Invalid Baseline Value::Metric '{metric_name}' in"
          f" baseline for '{config_id}' has non-numeric 'baseline_ms' or"
          " 'threshold'. Skipping."
      )
      continue

    # Extract the actual metric value from results.json
    actual_metric_entry = actual_metrics_container.get(metric_name)

    if not actual_metric_entry or "value_ms" not in actual_metric_entry:
      summary_messages.append(
          f"Metric '{metric_name}': Actual value or 'value_ms' key not found in"
          " results, or not a dictionary. Skipping."
      )
      # For debugging:
      # available_keys = (
      #     list(actual_metrics_container.keys())
      #     if actual_metrics_container
      #     else "None"
      # )
      # summary_messages.append(
      #   f"  Available metric keys in results 'metrics' object:"
      #   f" {available_keys}"
      # )
      continue

    try:
      actual_value_ms = float(actual_metric_entry["value_ms"])
    except (ValueError, TypeError):
      summary_messages.append(
          f"Metric '{metric_name}': Actual value"
          f" '{actual_metric_entry['value_ms']}' is not a valid number."
          " Skipping."
      )
      continue

    summary_messages.append(f"\nComparing metric: {metric_name}")
    summary_messages.append(f"  Actual Value: {actual_value_ms:.3f} ms")
    summary_messages.append(f"  Baseline Value: {baseline_value_ms:.3f} ms")
    summary_messages.append(
        f"  Allowed Threshold: {threshold_percentage*100:.1f}%"
    )

    # Higher value is worse for time-based metrics
    allowed_upper_bound = baseline_value_ms * (1.0 + threshold_percentage)
    summary_messages.append(
        "  Allowed Upper Bound (Baseline * (1 + Threshold)):"
        f" {allowed_upper_bound:.3f} ms"
    )

    if actual_value_ms > allowed_upper_bound:
      percentage_diff = 0.0
      if (
          abs(baseline_value_ms) > 1e-9
      ):  # Avoid division by zero for very small baselines
        percentage_diff = (
            (actual_value_ms - baseline_value_ms) / baseline_value_ms
        ) * 100.0
      elif (
          actual_value_ms > 0
      ):  # If baseline is effectively zero, any positive value is infinitely
        # worse.
        percentage_diff = float("inf")

      # Use GitHub Actions error annotation for better visibility
      error_title = f"REGRESSION: {metric_name}"
      error_details = (
          f"Value {actual_value_ms:.3f} ms is {percentage_diff:.2f}% worse than"
          f" baseline {baseline_value_ms:.3f} ms. Exceeds threshold of"
          f" {threshold_percentage*100:.1f}% (max allowed:"
          f" {allowed_upper_bound:.3f} ms)."
      )
      summary_messages.append(
          "  ::error"
          f" file={results_json_file},title={error_title}::{error_details}"
      )
      regressions_found = True
    else:
      summary_messages.append(
          f"  Metric '{metric_name}' is within threshold. PASSED."
      )
  return regressions_found, summary_messages


def main():
  parser = argparse.ArgumentParser(
      description=(
          "Compare benchmark results to baselines and fail on regression."
      ),
      formatter_class=argparse.RawTextHelpFormatter,
  )  # For better help text formatting
  parser.add_argument(
      "--results-json-file",
      required=True,
      help=(
          "Path to the benchmark results JSON file (e.g., output/results.json)."
      ),
  )
  parser.add_argument(
      "--baseline-yaml-file",
      required=True,
      help=(
          "Path to the baseline YAML file (e.g.,"
          " xla/tools/benchmarks/baseline/presubmit_baseline.yml)."
      ),
  )
  parser.add_argument(
      "--config-id",
      required=True,
      help=(
          "The configuration ID for the current benchmark run. \nThis ID must"
          " exactly match a top-level key in the baseline YAML file. \nExample:"
          " 'gemma3_1b_flax_call_gpu_b200_1_host_1_device'"
      ),
  )

  args = parser.parse_args()

  print("--- Benchmark Baseline Comparison ---")
  print(f"Results JSON: {args.results_json_file}")
  print(f"Baseline YAML: {args.baseline_yaml_file}")
  print(f"Config ID for Baseline Lookup: {args.config_id}")

  if not os.path.exists(args.results_json_file):
    print(
        f"::error file={args.results_json_file}::Results JSON file not found."
    )
    sys.exit(1)

  if not os.path.exists(args.baseline_yaml_file):
    print(
        f"::notice file={args.baseline_yaml_file}::Baseline YAML file not"
        " found. Skipping comparison."
    )
    # Exiting 0 because no baseline means no regression to detect.
    # If a baseline is mandatory, this could be sys.exit(1).
    sys.exit(0)

  results_data = load_results_data(args.results_json_file)
  print("\nLoaded Results Data:")
  print(json.dumps(results_data, indent=2))

  baseline_data_full = load_baseline_data(args.baseline_yaml_file)
  config_baselines = validate_baseline_data(
      baseline_data_full, args.config_id, args.baseline_yaml_file
  )
  print("\nLoaded Baseline Data for this config_id:")
  print(json.dumps(config_baselines, indent=2))

  regressions_found, summary_messages = compare_metrics(
      results_data,
      config_baselines,
      args.results_json_file,
      args.config_id,
  )

  print("\n".join(summary_messages))

  if regressions_found:
    print(
        "\n::error::One or more benchmark metrics regressed beyond the allowed"
        " threshold. Failing the check."
    )
    sys.exit(1)  # Exit with error code 1 to fail the GitHub Actions step
  else:
    print(
        "\nAll benchmark metrics are within allowed thresholds (or no"
        " applicable baselines found)."
    )
    sys.exit(0)  # Exit with success


if __name__ == "__main__":
  main()

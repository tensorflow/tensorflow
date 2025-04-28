# Copyright 2025 The OpenXLA Authors.
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
# ==============================================================================
"""Generates GitHub Actions matrix JSON for XLA benchmarks based on a registry file."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from typing import Any, Dict, List, Tuple

from google.protobuf import text_format

from xla.tools.benchmarks.proto import benchmark_config_pb2

# --- Mapping Logic ---
HARDWARE_TO_RUNNER_LABEL: Dict[str, str] = {
    "CPU_X86": "linux-x86-n2-128",
    "CPU_ARM64": "linux-arm64-c4a-64",
    "GPU_L4": "linux-x86-g2-16-l4-1gpu",
    "GPU_B200": "linux-x86-a4-224-b200-1gpu",
    "GPU_L4_1H_4D": "linux-x86-g2-48-l4-4gpu",
}

HARDWARE_TO_CONTAINER_IMAGE: Dict[str, str] = {
    "CPU_X86": (
        "us-central1-docker.pkg.dev/tensorflow-sigs/tensorflow/ml-build:latest"
    ),
    "CPU_ARM64": (
        "us-central1-docker.pkg.dev/tensorflow-sigs/tensorflow/ml-build-arm64:latest"
    ),
    "GPU_L4": (
        "us-central1-docker.pkg.dev/tensorflow-sigs/tensorflow/ml-build-cuda12.8-cudnn9.8:latest"
    ),
    "GPU_B200": (
        "us-central1-docker.pkg.dev/tensorflow-sigs/tensorflow/ml-build-cuda12.8-cudnn9.8:latest"
    ),
    "GPU_L4_1H_4D": (
        "us-central1-docker.pkg.dev/tensorflow-sigs/tensorflow/ml-build-cuda12.8-cudnn9.8:latest"
    ),
}

RUN_FREQUENCY_MAP: Dict[str, benchmark_config_pb2.RunFrequency] = {
    "presubmit": benchmark_config_pb2.PRESUBMIT,
    "postsubmit": benchmark_config_pb2.POSTSUBMIT,
    "scheduled": benchmark_config_pb2.SCHEDULED,
    "manual": benchmark_config_pb2.MANUAL,
}

TARGET_METRIC_MAP: Dict[str, List[benchmark_config_pb2.TargetMetric]] = {
    "CPU_X86": [benchmark_config_pb2.CPU_TIME],
    "CPU_ARM64": [benchmark_config_pb2.CPU_TIME],
    "GPU_L4": [
        benchmark_config_pb2.GPU_DEVICE_TIME,
        benchmark_config_pb2.GPU_DEVICE_MEMCPY_TIME,
    ],
    "GPU_B200": [
        benchmark_config_pb2.GPU_DEVICE_TIME,
        benchmark_config_pb2.GPU_DEVICE_MEMCPY_TIME,
    ],
    "GPU_L4_1H_4D": [
        benchmark_config_pb2.GPU_DEVICE_TIME,
        benchmark_config_pb2.GPU_DEVICE_MEMCPY_TIME,
    ],
}
# --- End Mapping Logic ---


def _generate_config_id(config: benchmark_config_pb2.BenchmarkConfig) -> str:
  """Generates a unique, filesystem-friendly ID for the benchmark configuration.

  Format: {name}_{hardware}_{num_hosts}h_{num_devices}d, e.g.
  "gemma2_2b_keras_jax_x86_1h_1d"

  Args:
    config: The benchmark configuration.

  Returns:
    A unique, filesystem-friendly ID for the benchmark configuration.
  """
  hw_enum_name = benchmark_config_pb2.HardwareCategory.Name(
      config.hardware_category
  )
  # Use lowercase and remove prefix for brevity
  hw_short = hw_enum_name.replace("GPU_", "").replace("CPU_", "").lower()
  topo = config.topology
  topo_str = f"{topo.num_hosts}h_{topo.num_devices_per_host}d"
  # Sanitize name slightly (replace spaces, etc., although proto name
  # shouldn't have them)
  sanitized_name = config.name.replace(" ", "_").replace("/", "_")
  return f"{sanitized_name}_{hw_short}_{topo_str}"


def _get_runner_info(
    config: benchmark_config_pb2.BenchmarkConfig,
) -> Tuple[str | None, str | None]:
  """Maps BenchmarkConfig hardware/topology to GHA runner label and container."""
  hw_enum_name = benchmark_config_pb2.HardwareCategory.Name(
      config.hardware_category
  )
  topology = config.topology

  mapping_key = hw_enum_name  # Default to just hardware type

  runner_label = HARDWARE_TO_RUNNER_LABEL.get(mapping_key)
  container_image = HARDWARE_TO_CONTAINER_IMAGE.get(mapping_key)

  if not runner_label or not container_image:
    print(
        f"Warning: No runner/container mapping for config '{config.name}' "
        f"with hardware '{hw_enum_name}' and topology "
        f"({topology.num_hosts}h/{topology.num_devices_per_host}d). "
        f"Mapping key used: '{mapping_key}'. Skipping config.",
        file=sys.stderr,
    )
    return None, None

  return runner_label, container_image


def _parse_registry(
    registry_path: str,
) -> benchmark_config_pb2.BenchmarkSuite | None:
  """Parses a TextProto registry file into a BenchmarkSuite proto."""
  suite = benchmark_config_pb2.BenchmarkSuite()
  try:
    with open(registry_path, "r") as f:
      content = f.read()
      text_format.Parse(content, suite)
    return suite
  except FileNotFoundError:
    print(f"Error: Registry file not found at {registry_path}", file=sys.stderr)
    return None
  except text_format.ParseError as e:
    print(
        f"Error parsing TextProto registry file {registry_path}: {e}",
        file=sys.stderr,
    )
    return None
  except Exception as e:  # pylint: disable=broad-exception-caught
    print(
        f"An unexpected error occurred while reading {registry_path}: {e}",
        file=sys.stderr,
    )
    return None


def generate_matrix(
    suite: benchmark_config_pb2.BenchmarkSuite,
) -> Dict[str, List[Dict[str, Any]]]:
  """Generates the GitHub Actions matrix dictionary."""
  matrix = {"include": []}

  for config in suite.configs:
    for run_frequency in config.run_frequencies:
      config_id = _generate_config_id(config)
      runner_label, container_image = _get_runner_info(config)

      if runner_label and container_image:
        hlo_location = (
            config.hlo_path if config.hlo_path else config.hlo_gcs_bucket_path
        )
        is_gcs_hlo = True if config.hlo_gcs_bucket_path else False

        run_frequency_name = benchmark_config_pb2.RunFrequency.Name(
            run_frequency
        )
        xla_flags_str = list(config.xla_compilation_flags)
        runtime_flags_str = list(config.runtime_flags)
        topology_dict = {
            "multi_host": config.topology.multi_host,
            "multi_device": config.topology.multi_device,
            "num_hosts": config.topology.num_hosts,
            "num_devices_per_host": config.topology.num_devices_per_host,
        }

        hardware_category_name = benchmark_config_pb2.HardwareCategory.Name(
            config.hardware_category
        )
        target_metrics_str = [
            benchmark_config_pb2.TargetMetric.Name(metric)
            for metric in TARGET_METRIC_MAP.get(hardware_category_name)
        ]
        github_labels_str = list(config.github_labels)

        # Serialize lists and dicts for GH Actions
        xla_flags_json = json.dumps(xla_flags_str)
        runtime_flags_json = json.dumps(runtime_flags_str)
        topology_json = json.dumps(topology_dict)
        github_labels_json = json.dumps(github_labels_str)

        matrix["include"].append({
            "config_id": config_id,
            "benchmark_name": config.name,
            "run_frequency": run_frequency_name,
            "runner_label": runner_label,
            "container_image": container_image,
            "hlo_location": hlo_location,
            "is_gcs_hlo": is_gcs_hlo,
            "target_metrics": target_metrics_str,
            "xla_compilation_flags": xla_flags_json,
            "runtime_flags": runtime_flags_json,
            "required_hardware_category": hardware_category_name,
            "topology": topology_json,
            "github_labels": github_labels_json,
        })

  return matrix


def _resolve_registry_path(registry_path: pathlib.Path) -> pathlib.Path:
  """Resolves the registry path, returning an absolute path if found.

  Resolution Order:
  1. If the input path is absolute and exists, return it.
  2. If relative, check relative to BUILD_WORKSPACE_DIRECTORY (if set).

  Args:
    registry_path: The path to the registry file (pathlib.Path).

  Returns:
    The absolute path to the registry file.

  Raises:
    FileNotFoundError: If the path is absolute but doesn't exist, or if it's
      relative and cannot be found in the workspace or relative to CWD.
  """
  try:
    # 1. Check if absolute
    if registry_path.is_absolute():
      if registry_path.exists():
        print(
            f"Registry path is absolute and exists: {registry_path}",
            file=sys.stderr,
        )
        return registry_path
      else:
        # Absolute path provided, but file doesn't exist at that location.
        raise FileNotFoundError(
            f"Absolute registry path specified but not found: {registry_path}"
        )

    # --- Path is Relative ---
    print(
        f"Registry file path '{registry_path}' is relative. Attempting"
        " resolution...",
        file=sys.stderr,
    )

    # 2. Check relative to BUILD_WORKSPACE_DIRECTORY (if set)
    build_workspace_dir = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
    if build_workspace_dir:
      try:
        workspace_path = (
            pathlib.Path(build_workspace_dir) / registry_path
        ).resolve()  # resolve() makes it absolute and cleans '..' etc.
        print(f"Checking workspace path: {workspace_path}", file=sys.stderr)
        if workspace_path.exists():
          print(
              f"Found registry file in workspace: {workspace_path}",
              file=sys.stderr,
          )
          return workspace_path
        else:
          print(
              f"Registry file not found at workspace path: {workspace_path}",
              file=sys.stderr,
          )
          # Continue to check relative to CWD...
      except Exception as e:  # pylint: disable=broad-exception-caught
        # Handle potential errors during workspace path resolution
        print(f"Warning: Error resolving workspace path: {e}", file=sys.stderr)
        # Continue to check relative to CWD...
    else:
      print(
          "BUILD_WORKSPACE_DIRECTORY not set, skipping workspace check.",
          file=sys.stderr,
      )

    # --- If we reach here, the file was not found ---
    raise FileNotFoundError(
        f"Registry file '{registry_path}' not found. Tried absolute, relative"
        " to workspace (if BUILD_WORKSPACE_DIRECTORY was set)."
    )

  # Catch potential fundamental issues with the input path object itself
  except OSError as e:
    raise ValueError(f"Invalid path provided: {registry_path}") from e


def main():
  parser = argparse.ArgumentParser(
      description=(
          "Generate GitHub Actions matrix JSON for XLA benchmarks based on a"
          " registry file."
      ),
      formatter_class=argparse.RawTextHelpFormatter,
  )

  parser.add_argument(
      "--registry_file",
      required=True,
      type=pathlib.Path,
      help="Path to the benchmark registry file (TextProto format).",
  )

  args = parser.parse_args()
  registry_path: pathlib.Path = args.registry_file
  resolved_registry_path = _resolve_registry_path(registry_path)
  print(f"Using final registry path: {resolved_registry_path}", file=sys.stderr)

  suite = _parse_registry(resolved_registry_path)
  if not suite:
    sys.exit(1)

  matrix_output = generate_matrix(suite)

  # Output JSON matrix to stdout
  print(json.dumps(matrix_output, indent=2))


if __name__ == "__main__":
  main()

# Copyright 2026 The OpenXLA Authors
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

"""Utilities for generating, logging, and saving benchmark results tables."""

from collections.abc import Sequence
import dataclasses
import os
from typing import Any

from absl import logging
import numpy as np
import pandas as pd

from xla.benchmarks.core import benchmark
from xla.benchmarks.jax_microbenchmarks import jax_profiler_utils


def config_to_dict(cfg: Any) -> dict[str, Any]:
  """Converts a config dataclass to a dictionary with formatted values."""
  if dataclasses.is_dataclass(cfg):
    return dataclasses.asdict(cfg)
  raise ValueError(f"Expected dataclass or dict, got {type(cfg)}")


def extract_metrics(
    profiler_results: Sequence[jax_profiler_utils.JaxProfilerResult | None],
) -> dict[str, Any]:
  """Extracts average latency in microseconds and FLOPS from profiler results."""
  all_runtimes = []
  flops_values = []
  for res in profiler_results:
    if res is not None:
      if res.runtimes_us:
        all_runtimes.extend(res.runtimes_us)
      if res.flops:
        flops_values.append(res.flops)

  avg_latency_us = float(np.mean(all_runtimes)) if all_runtimes else None
  flops = float(np.mean(flops_values)) if flops_values else None

  return {
      "latency_us": avg_latency_us,
      "flops": flops,
  }


def create_results_table(
    results: Sequence[
        tuple[
            benchmark.BenchmarkConfig,
            Sequence[jax_profiler_utils.JaxProfilerResult | None],
        ]
    ],
) -> pd.DataFrame:
  """Generates a pandas DataFrame table of results for a benchmark run.

  Args:
    results: Sequence of (config, profiler_results) pairs.

  Returns:
    A DataFrame whose columns are the fields of the config object followed by
    latency_us and flops.
  """
  rows = []
  for cfg, prof_results in results:
    row = cfg.as_dict()
    metrics = extract_metrics(prof_results)
    row.update(metrics)
    rows.append(row)
  return pd.DataFrame(rows)


def log_results_table(
    benchmark_name: str,
    df: pd.DataFrame,
) -> None:
  """Logs the benchmark results table."""
  logging.info(
      "Benchmark Results for '%s':\n%s",
      benchmark_name,
      df.to_string(index=False),
  )


def write_results_to_csv(
    df: pd.DataFrame,
    csv_path: str,
) -> None:
  """Writes a results DataFrame to a CSV file at csv_path."""
  parent_dir = os.path.dirname(os.path.abspath(csv_path))
  if parent_dir:
    os.makedirs(parent_dir, exist_ok=True)
  df.to_csv(csv_path, index=False)
  logging.info("Wrote results to %s", csv_path)


def save_all_results_to_csv(
    results_by_benchmark: dict[str, pd.DataFrame],
    csv_path: str,
) -> None:
  """Writes all benchmark results tables to CSV file(s).

  Args:
    results_by_benchmark: Dictionary mapping benchmark names to DataFrames.
    csv_path: Destination path. If a directory or path ending with separator,
      writes `<dir>/<benchmark_name>.csv`. If a file path, only one benchmark
      should be provided.
  """
  if not csv_path or not results_by_benchmark:
    return

  is_dir = os.path.isdir(csv_path) or csv_path.endswith(("/", "\\"))
  if is_dir:
    for name, df in results_by_benchmark.items():
      dest = os.path.join(csv_path, f"{name}.csv")
      write_results_to_csv(df, dest)
  elif len(results_by_benchmark) == 1:
    _, df = next(iter(results_by_benchmark.items()))
    write_results_to_csv(df, csv_path)
  else:
    raise ValueError(
        "Multiple benchmarks provided but csv_path is not a directory."
    )

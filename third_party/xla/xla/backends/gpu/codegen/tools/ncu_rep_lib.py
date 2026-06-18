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
"""Library to work with ncu-rep CSV output.

Separate library from ncu_rep_main.py to enable unit tests.
"""

import csv
import itertools
import json
import logging
import re
from typing import TextIO
from absl import app

KERNEL_NAME_FIELD = "Kernel Name"
KERNEL_ID_FIELD = "ID"


def get_metrics_by_kernel(
    rows: list[list[str]],
) -> list[dict[str, tuple[str, str]]]:
  """Converts ncu-rep table to a dictionary of metrics by kernel.

  Args:
    rows: ncu-rep table rows

  Returns:
    dictionary of metrics by kernel
  """
  name_index = {}
  units = rows[1]
  for i, name in enumerate(rows[0]):
    name_index[name] = i
  results = []
  for kernel in rows[2:]:
    values = {}
    for idx, name in enumerate(rows[0]):
      values[name] = (kernel[idx], units[idx])
    results.append(values)
  return results


def aggregate(metric_values: list[float], metric_name: str):
  """Aggregates metric values using a function based on the metric name.

  If metric name does not match any of the known patterns, the first value from
  the input list is returned.

  Args:
    metric_values: list of metric values, floats
    metric_name: metric name

  Returns:
    aggregated metric value
  """
  if str.endswith(metric_name, ".max"):
    return max(metric_values)
  if str.endswith(metric_name, ".min"):
    return min(metric_values)
  if str.endswith(metric_name, ".sum"):
    return sum(metric_values)
  return metric_values[0]


def aggregate_kernel_metrics(
    metrics: list[str], kernel_metrics: list[dict[str, tuple[str, str]]]
) -> list[list[str]]:
  """Aggregates and returns the metrics for the given kernels.

  Args:
    metrics: list of metrics names to print
    kernel_metrics: dictionary of metrics by kernel

  Returns:
    list of rows [name, value, unit] per metric.
  """
  if not kernel_metrics:
    raise app.UsageError("no metrics found")
  results: dict[str, tuple[list[float], str]] = {}  # name -> (float[], unit)
  for vals in kernel_metrics:
    for name in metrics:
      if name not in vals:
        raise app.UsageError(f"metric '{name}' is not found")
      value, unit = vals[name]
      if name not in results:
        results[name] = ([], unit)
      if results[name][1] != unit:
        # That should not happen with `--print-units base` but left as a
        # safety check.
        raise app.UsageError(f"unit mismatch for metric '{name}'")
      # Replace ',' in value to parse it as a float. It is printed in
      # en_US.UTF-8 locale but we don't want to change the runtime locale just
      # for this as that might affect downstream.
      results[name][0].append(float(value.replace(",", "")))
  kernel_metrics = []
  for name, (values, unit) in results.items():
    a = aggregate(values, name)
    if round(a) == a:
      kernel_metrics.append([name, f"{round(a)}", unit])
    else:
      kernel_metrics.append([name, f"{round(a, 2)}", unit])
  return kernel_metrics


def filter_kernels(
    kernels: list[dict[str, tuple[str, str]]], condition: str
) -> list[dict[str, tuple[str, str]]]:
  """Filters kernels by a condition.

  Args:
    kernels: list of kernel tuples, extracted from ncu-rep CSV
    condition: filter condition. Supported filter expressions: 'id:<value>' -
      kernel with an ID 'name:<regex>' - kernel with a name matching the regex
      'after:<filter>' - kernels after the last kernel matching the filter.

  Returns:
    matching kernels
  """
  if condition.startswith("id:"):
    i = condition.removeprefix("id:")
    return [v for v in kernels if v[KERNEL_ID_FIELD][0] == i]
  if condition.startswith("name:"):
    r = condition.removeprefix("name:")
    return [v for v in kernels if re.search(r, v[KERNEL_NAME_FIELD][0])]
  if condition.startswith("after:"):
    r = condition.removeprefix("after:")
    sub = filter_kernels(kernels, r)
    if not sub:
      logging.warning("no kernels matched '%s', 'after:' has no effect", r)
      return kernels
    after_id = sub[-1][KERNEL_ID_FIELD][0]
    return list(
        itertools.dropwhile(
            lambda v: v[KERNEL_ID_FIELD][0] != after_id, kernels
        )
    )[1:]
  raise app.UsageError(f"unsupported filter: {condition}")


def write_metrics_markdown(out: TextIO, metrics: list[list[str]]):
  """Formats metrics in markdown."""
  name_width = max(len(m[0]) for m in metrics)
  value_width = max(max(len(m[1]) for m in metrics), len("value"))
  unit_width = max(max(len(m[2]) for m in metrics), len("unit"))
  out.write(
      f"{'Metric'.ljust(name_width)} | {'Value'.rjust(value_width)} | Unit\n"
  )
  out.write("-" * name_width)
  out.write("-|-")
  out.write("-" * value_width)
  out.write("-|-")
  out.write("-" * unit_width + "\n")
  for name, value, unit in metrics:
    out.write(
        f"{name.ljust(name_width)} | {value.rjust(value_width)} | {unit}\n"
    )


def write_metrics_csv(out: TextIO, metrics: list[list[str]]):
  """Formats metrics in csv."""
  writer = csv.writer(out, lineterminator="\n")
  writer.writerow(["metric", "value", "unit"])
  writer.writerows(metrics)


def write_metrics_json(out: TextIO, metrics: list[list[str]]):
  """Formats metrics in JSON."""
  data = {}
  for name, value, unit in metrics:
    data[name] = {"value": value, "unit": unit}
  json.dump(data, out, sort_keys=True)
  out.write("\n")


def write_metrics_raw(out: TextIO, metrics: list[list[str]]):
  """Formats metrics in raw."""
  for _, value, unit in metrics:
    out.write(f"{value} {unit}\n")

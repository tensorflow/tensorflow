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
import json
from typing import TextIO
from absl import app


def get_metrics_by_kernel(
    rows: list[list[str]],
) -> dict[str, dict[str, tuple[str, str]]]:
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
  results = {}
  for kernel in rows[2:]:
    values = {}
    for idx, name in enumerate(rows[0]):
      values[name] = (kernel[idx], units[idx])
    kernel_name = values["Kernel Name"][0]
    results[kernel_name] = values
  return results


def get_kernel_metrics_rows(
    metrics: list[str],
    all_metrics: dict[str, dict[str, tuple[str, str]]],
    kernel_name: str,
) -> list[list[str]]:
  """Returns the metrics to print for the given kernel.

  Args:
    metrics: list of metrics names to print
    all_metrics: dictionary of metrics by kernel, extracted from ncu-rep table
    kernel_name: kernel name to print, returns first kernel if empty

  Returns:
    list of rows [name, value, unit] per metric.
  """
  if not all_metrics:
    raise app.UsageError("no metrics found")
  for kernel, vals in all_metrics.items():
    if kernel_name and kernel != kernel_name:
      continue
    result = []
    for name in metrics:
      if name not in vals:
        raise app.UsageError(f"metric '{name}' is not found")
      result.append([name, vals[name][0], vals[name][1]])
    return result
  raise app.UsageError(f"kernel '{kernel_name}' is not found")


def write_metrics_markdown(out: TextIO, metrics: list[list[str]]):
  """Formats metrics in markdown."""
  name_width = max(len(m[0]) for m in metrics)
  value_width = max(max(len(m[1]) for m in metrics), len("value"))
  unit_width = max(max(len(m[2]) for m in metrics), len("unit"))
  out.write(
      f"{'Metric'.ljust(name_width)} | {'Value'.rjust(value_width)} | Unit\n"
  )
  out.write(
      f"{'-' * name_width }-|-{'-' * value_width }-|-{'-' * unit_width }\n"
  )
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

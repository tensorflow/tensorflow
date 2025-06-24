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
"""Print metrics from ncu-rep file.

Usage:
  ncu_rep <ncu-rep-file> [--list_kernels] [--list_metrics]
    [--filter <filter kernels>]
    [-f <format>] [-k <kernel name>]
    [-m metric1] [-m metric2]f metrics: print all metric names
"""

from collections.abc import Sequence
import csv
import logging
import os
import shutil
import subprocess
import sys
from absl import app
from absl import flags
from xla.backends.gpu.codegen.tools import ncu_rep_lib

_METRICS = flags.DEFINE_multi_string(
    "m",
    [
        "gpu__time_duration.sum",
        "sm__cycles_elapsed.max",
        "dram__bytes_read.sum",
        "dram__bytes_write.sum",
        "launch__registers_per_thread",
    ],
    "Names of metrics to print",
)
_FORMAT = flags.DEFINE_enum(
    "f",
    "md",
    ["md", "csv", "json", "raw"],
    "Output format: md (default), csv, json or plain text",
)
_KERNEL_FILTER = flags.DEFINE_list(
    "filter",
    None,
    "kernel filter: comma-separated list of kernel predicates:\n-"
    " 'name:<regex>' matches part of the name, case sensitive. Use"
    " 'name:^<regex>$' to match the whole string\n- 'id:<number>' - numeric"
    " kernel id\n- 'after:<matcher>' - match all kernels after the all matces"
    " of the matcher\nFilters are applied in order they are specified",
)
_LIST_KERNELS = flags.DEFINE_bool(
    "list_kernels", None, "print kernel names and exit", required=False
)
_LIST_METRICS = flags.DEFINE_bool(
    "list_metrics", None, "print metric names and exit", required=False
)

ncu_bin = shutil.which("ncu")
if not ncu_bin:
  ncu_bin = "/usr/local/cuda/bin/ncu"
logging.info("ncu binary: %s", ncu_bin)


def main(argv: Sequence[str]) -> None:
  if len(argv) != 2:
    raise app.UsageError("provide .ncu-rep file path")
  input_file_name = argv[1]
  if not os.path.exists(input_file_name):
    raise app.UsageError(f"file '{input_file_name}' does not exist")
  cmd = [
      ncu_bin,
      "-i",
      input_file_name,
      "--csv",
      "--print-units",
      "base",
      "--page",
      "raw",
  ]
  env_with_locale = os.environ.copy()
  # Force locale to en_US.UTF-8 to get consistent output.
  env_with_locale["LC_ALL"] = "en_US.UTF-8"
  # env_with_locale["LC_ALL"] = "de_DE.UTF-8"
  out = subprocess.check_output(cmd, text=True, env=env_with_locale).strip()
  rows = list(csv.reader(out.splitlines()))
  name_index = {}
  for i, name in enumerate(rows[0]):
    name_index[name] = i

  if _LIST_METRICS.value:
    for name in rows[0]:
      print(name)
    return

  all_kernels = ncu_rep_lib.get_metrics_by_kernel(rows)
  filtered_kernels = all_kernels
  for f in _KERNEL_FILTER.value or []:
    filtered_kernels = ncu_rep_lib.filter_kernels(filtered_kernels, f)
  if not filtered_kernels:
    raise app.UsageError(
        "No kernels matched the filter, use --list_kernels without --filter to"
        " see all kernels"
    )

  if _LIST_KERNELS.value:
    for row in filtered_kernels:
      print(
          row[ncu_rep_lib.KERNEL_ID_FIELD][0],
          row[ncu_rep_lib.KERNEL_NAME_FIELD][0],
      )
    return

  if len(filtered_kernels) > 1:
    sys.stderr.write(
        f"aggregating {len(filtered_kernels)} kernels\n",
    )

  metrics = ncu_rep_lib.aggregate_kernel_metrics(
      _METRICS.value, filtered_kernels
  )

  fmt = _FORMAT.value
  if fmt == "csv":
    ncu_rep_lib.write_metrics_csv(sys.stdout, metrics)
  elif fmt == "json":
    ncu_rep_lib.write_metrics_json(sys.stdout, metrics)
  elif fmt == "raw":
    ncu_rep_lib.write_metrics_raw(sys.stdout, metrics)
  else:
    ncu_rep_lib.write_metrics_markdown(sys.stdout, metrics)


if __name__ == "__main__":
  app.run(main)

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
  ncu_rep -i <ncu-rep-file> [metrics|kernels|value]
    [-f <format>] [-k <kernel name>]
    [-m metric1] [-m metric2]
  metrics: print all metric names
  kernels: print all kernel names
  value (default): print values of metrics as in -m
"""

from collections.abc import Sequence
import csv
import logging
import shutil
import subprocess
import sys
from absl import app
from absl import flags
from xla.backends.gpu.codegen.tools import ncu_rep_lib

_INPUT_FILE = flags.DEFINE_string(
    "i", None, "Input .ncu-rep file", required=False
)
_METRICS = flags.DEFINE_multi_string(
    "m",
    [
        "gpu__time_duration.sum",
        "sm__cycles_elapsed.max",
        "dram__bytes_read.sum",
        "dram__bytes_write.sum",
        "launch__registers_per_thread",
    ],
    "Input .ncu-rep file",
)
_FORMAT = flags.DEFINE_enum(
    "f",
    "md",
    ["md", "csv", "json", "raw"],
    "Output format: md (default), csv, or json",
)
_KERNEL = flags.DEFINE_string(
    "k",
    None,
    "kernel to print (prints first kernel if empty)",
)

ncu_bin = shutil.which("ncu")
if not ncu_bin:
  ncu_bin = "/usr/local/cuda/bin/ncu"
logging.info("ncu binary: %s", ncu_bin)


def main(argv: Sequence[str]) -> None:
  input_name = _INPUT_FILE.value
  if not input_name:
    # We can't use required=True due to unit tests.
    raise app.UsageError("input file (-i) is required")
  cmd = [ncu_bin, "-i", input_name, "--csv", "--page", "raw"]
  out = subprocess.check_output(cmd, text=True).strip()
  rows = list(csv.reader(out.splitlines()))
  name_index = {}
  for i, name in enumerate(rows[0]):
    name_index[name] = i

  op = argv[1] if len(argv) > 1 else "value"
  if op == "metrics":
    for name in rows[0]:
      print(name)
    return

  metrics_by_kernel = ncu_rep_lib.get_metrics_by_kernel(rows)

  if op == "kernels":
    for name in metrics_by_kernel:
      print(name)
    return

  metrics = ncu_rep_lib.get_kernel_metrics_rows(
      _METRICS.value, metrics_by_kernel, _KERNEL.value
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

# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Parses results from run_onednn_benchmarks.sh.

Example results:
               Model,  Batch,        Vanilla,         oneDNN,    Speedup
          bert-large,      1,       984508.0,      1511545.0,       1.54
          bert-large,     16,              ?,              ?,          ?
           inception,      1,        13720.0,        12859.0,       0.94
           inception,     16,       162221.0,       137648.0,       0.85
        mobilenet-v1,      1,        18052.0,        19196.0,       1.06
        mobilenet-v1,     16,       140987.0,       143874.0,       1.02
       resnet50_v1-5,      1,        46919.0,        59567.0,       1.27
       resnet50_v1-5,     16,       557088.0,      1128931.0,       2.03
    ssd-mobilenet-v1,      1,        35998.0,        27543.0,       0.77
    ssd-mobilenet-v1,     16,       365288.0,       235566.0,       0.64
        ssd-resnet34,      1,              ?,     22706217.0,          ?
        ssd-resnet34,     16,              ?,    229083059.0,          ?
"""

import enum
import re
import sys

db = dict()
models = set()
batch_sizes = set()
State = enum.Enum("State", "FIND_CONFIG_OR_MODEL FIND_RUNNING_TIME")


def parse_results(lines):
  """Parses benchmark results from run_onednn_benchmarks.sh.

  Stores results in a global dict.

  Args:
    lines: Array of strings corresponding to each line of the output from
      run_onednn_benchmarks.sh

  Raises:
    RuntimeError: If the program reaches an unknown state.
  """
  idx = 0
  batch, onednn, model = None, None, None
  state = State.FIND_CONFIG_OR_MODEL
  while idx < len(lines):
    if state is State.FIND_CONFIG_OR_MODEL:
      config = re.match(
          r"\+ echo 'BATCH=(?P<batch>[\d]+), ONEDNN=(?P<onednn>[\d]+)",
          lines[idx])
      if config:
        batch = int(config.group("batch"))
        onednn = int(config.group("onednn"))
        batch_sizes.add(batch)
      else:
        model_re = re.search(r"tf-graphs\/(?P<model>[\w\d_-]+).pb", lines[idx])
        assert model_re
        model = model_re.group("model")
        models.add(model)
        state = State.FIND_RUNNING_TIME
    elif state is State.FIND_RUNNING_TIME:
      match = re.search(r"no stats: (?P<avg>[\d.]+)", lines[idx])
      state = State.FIND_CONFIG_OR_MODEL
      if match:
        avg = float(match.group("avg"))
        key = (model, batch, onednn)
        assert None not in key
        db[key] = avg
      else:
        # Some models such as ssd-resnet34 can't run on CPU with vanilla TF and
        # won't have results. This line contains either a config or model name.
        continue
    else:
      raise RuntimeError("Reached the unreachable code.")
    idx = idx + 1


def main():
  filename = sys.argv[1]
  with open(filename, "r") as f:
    lines = f.readlines()
  parse_results(lines)
  print("%20s, %6s, %14s, %14s, %10s" %
        ("Model", "Batch", "Vanilla", "oneDNN", "Speedup"))
  for model in sorted(models):
    for batch in sorted(batch_sizes):
      key = (model, batch, 0)
      eigen = db[key] if key in db else "?"
      key = (model, batch, 1)
      onednn = db[key] if key in db else "?"
      speedup = "%10.2f" % (onednn / eigen) if "?" not in (eigen,
                                                           onednn) else "?"
      print("%20s, %6d, %14s, %14s, %10s" %
            (model, batch, str(eigen), str(onednn), speedup))


if __name__ == "__main__":
  main()

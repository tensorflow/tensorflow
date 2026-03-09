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
"""Parses benchmark results.json and dumps metrics to TensorBoard.

See .github/workflows/benchmarks for more details.
"""

import argparse
import json
import os
import sys
import time

from tensorboard.compat.proto import event_pb2
from tensorboard.compat.proto import summary_pb2
from tensorboard.summary.writer.event_file_writer import EventFileWriter


def main():
  parser = argparse.ArgumentParser(
      description="Convert results.json to TensorBoard events."
  )
  parser.add_argument(
      "--results-json", required=True, help="Path to results.json"
  )
  parser.add_argument(
      "--step", type=int, default=0, help="Global step for TensorBoard events"
  )
  args = parser.parse_args()

  results_file = args.results_json
  output_dir = os.environ.get("TENSORBOARD_OUTPUT_DIR")

  if not output_dir:
    print("::error::TENSORBOARD_OUTPUT_DIR environment variable must be set.")
    sys.exit(1)

  if not os.path.exists(results_file):
    print(f"::error::Results file '{results_file}' not found.")
    sys.exit(1)

  try:
    with open(results_file, "r") as f:
      data = json.load(f)
  except json.JSONDecodeError as e:
    print(f"::error::Failed to parse results JSON: {e}")
    sys.exit(1)
  except OSError as e:
    print(f"::error::Error reading results JSON: {e}")
    sys.exit(1)

  metrics = data.get("metrics", {})
  if not metrics:
    print(f"::warning::No metrics found in {results_file}.")
    return

  if not os.path.exists(output_dir):
    try:
      os.makedirs(output_dir)
    except OSError as e:
      print(f"::error::Failed to create output directory {output_dir}: {e}")
      sys.exit(1)

  print(f"Writing TensorBoard events to {output_dir}...")

  try:
    writer = EventFileWriter(output_dir)

    # Use current time for wall_time
    wall_time = time.time()
    step = args.step

    for metric_name, metric_data in metrics.items():
      val = metric_data.get("value")

      if val is None:
        continue

      try:
        val = float(val)
      except (ValueError, TypeError):
        print(f"::warning::Skipping non-numeric metric '{metric_name}': {val}")
        continue

      summary = summary_pb2.Summary(
          value=[summary_pb2.Summary.Value(tag=metric_name, simple_value=val)]
      )
      event = event_pb2.Event(wall_time=wall_time, step=step, summary=summary)
      writer.add_event(event)

    writer.close()
    print(f"Successfully wrote {len(metrics)} metrics to TensorBoard.")

  except (IOError, OSError, ValueError) as e:
    print(f"::error::Failed to write TensorBoard events: {e}")
    sys.exit(1)


if __name__ == "__main__":
  main()

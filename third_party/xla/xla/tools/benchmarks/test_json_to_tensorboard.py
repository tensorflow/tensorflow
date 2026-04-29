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
"""Tests for json_to_tensorboard.py."""

import json
import os
import shutil
import sys
import unittest

import json_to_tensorboard
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


class TestJsonToTensorBoard(unittest.TestCase):
  """Tests parsing of results.json and conversion to TensorBoard events."""

  def setUp(self):
    super().setUp()
    self.gpu_json_path = "gpu_results.json"
    self.cpu_json_path = "cpu_results.json"
    self.output_dir = "test_tb_output"

    self.gpu_data = {
        "metrics": {
            "GPU_DEVICE_TIME": {"value": 10.5, "unit": "ms"},
            "GPU_DEVICE_MEMCPY_TIME": {"value": 1.2, "unit": "ms"},
            "PEAK_GPU_MEMORY": {"value": 4.5, "unit": "GB"},
        }
    }
    self.cpu_data = {
        "metrics": {
            "WALL_TIME": {"value": 100.0, "unit": "ms"},
            "CPU_TIME": {"value": 85.0, "unit": "ms"},
        }
    }

    with open(self.gpu_json_path, "w") as f:
      json.dump(self.gpu_data, f)
    with open(self.cpu_json_path, "w") as f:
      json.dump(self.cpu_data, f)

    os.environ["TENSORBOARD_OUTPUT_DIR"] = self.output_dir

  def tearDown(self):
    for p in [self.gpu_json_path, self.cpu_json_path]:
      if os.path.exists(p):
        os.remove(p)
    if os.path.exists(self.output_dir):
      shutil.rmtree(self.output_dir)
    if "TENSORBOARD_OUTPUT_DIR" in os.environ:
      del os.environ["TENSORBOARD_OUTPUT_DIR"]
    super().tearDown()

  def run_script(self, json_path, step):
    """Helper to run the json_to_tensorboard script with specific arguments."""
    test_args = ["prog", "--results-json", json_path, "--step", str(step)]
    original_argv = sys.argv
    try:
      sys.argv = test_args
      json_to_tensorboard.main()
    finally:
      sys.argv = original_argv

  def test_gpu_metrics(self):
    """Verifies that GPU metrics are correctly converted to TensorBoard."""
    test_step = 10
    self.run_script(self.gpu_json_path, test_step)

    # Verify output directory was created
    self.assertTrue(os.path.exists(self.output_dir))

    # Find and load the event file
    ea = EventAccumulator(self.output_dir)
    ea.Reload()

    # Check scalar tags
    tags = ea.Tags()["scalars"]
    expected_tags = [
        "GPU_DEVICE_TIME (ms)",
        "GPU_DEVICE_MEMCPY_TIME (ms)",
        "PEAK_GPU_MEMORY (GB)",
    ]

    for tag in expected_tags:
      self.assertIn(tag, tags)
      events = ea.Scalars(tag)
      self.assertEqual(len(events), 1)
      self.assertEqual(events[0].step, test_step)

      # Verify value
      base_key = tag.split(" (")[0]
      self.assertAlmostEqual(
          events[0].value, self.gpu_data["metrics"][base_key]["value"]
      )

  def test_cpu_metrics(self):
    """Verifies that CPU metrics are correctly converted to TensorBoard."""
    test_step = 20
    self.run_script(self.cpu_json_path, test_step)

    # Verify output directory was created
    self.assertTrue(os.path.exists(self.output_dir))

    # Find and load the event file
    ea = EventAccumulator(self.output_dir)
    ea.Reload()

    # Check scalar tags
    tags = ea.Tags()["scalars"]
    expected_tags = ["WALL_TIME (ms)", "CPU_TIME (ms)"]

    for tag in expected_tags:
      self.assertIn(tag, tags)
      events = ea.Scalars(tag)
      self.assertEqual(len(events), 1)
      self.assertEqual(events[0].step, test_step)

      # Verify value
      base_key = tag.split(" (")[0]
      self.assertAlmostEqual(
          events[0].value, self.cpu_data["metrics"][base_key]["value"]
      )


if __name__ == "__main__":
  unittest.main()

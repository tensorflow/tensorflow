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

import json

from absl.testing import absltest
from absl.testing import parameterized
from google.protobuf import text_format
import more_itertools

from xla.tools.benchmarks.proto import benchmark_config_pb2
from xla.tools.benchmarks.utils import generate_benchmark_matrices as generate_matrices

# TextProto representation of the data in default_registry.yml
# NOTE: The original script expects TextProto, not YAML.
DEFAULT_REGISTRY_TEXTPROTO = """
configs {
  name: "gemma3_1b_flax_call"
  description: "Benchmarks Gemma3 1b in Flax using B200 GPUs."
  owner: "juliagmt-google@"
  hlo_gcs_bucket_path: "https://storage.googleapis.com/xla-benchmarking-temp/gemma3_1b_flax_call.hlo"
  model_source_info: "Gemma3 1B"
  hardware_category: GPU_B200
  topology { num_hosts: 1 num_devices_per_host: 1 }
  run_frequencies: POSTSUBMIT
  update_frequency_policy: QUARTERLY
  runtime_flags: "--num_repeat=5"
  github_labels: "blocking_presubmit_test"
}
configs {
  name: "gemma2_2b_keras_jax"
  description: "Gemma2 2B in Keras on x86 CPU."
  owner: "company-A@"
  hlo_path: "benchmarks/hlo/gemma2_2b_keras_jax.hlo"
  model_source_info: "Gemma2 2B"
  hardware_category: CPU_X86
  topology { num_hosts: 1 num_devices_per_host: 1 }
  # Note: Even though PEAK_CPU_MEMORY is in YAML, the script's TARGET_METRIC_MAP
  # only includes CPU_TIME for CPU_X86, so only CPU_TIME will appear in the matrix.
  # We include both here to test parsing.
  target_metrics: CPU_TIME
  target_metrics: PEAK_CPU_MEMORY
  run_frequencies: PRESUBMIT
  run_frequencies: POSTSUBMIT
  update_frequency_policy: QUARTERLY
  runtime_flags: "--num_repeat=5"
  github_labels: "blocking_presubmit_test"
}
"""


class GenerateBenchmarkMatricesTest(parameterized.TestCase):

  def test_parse_registry_success(self):
    """Tests successful parsing of a valid TextProto registry."""
    temp_file = self.create_tempfile(content=DEFAULT_REGISTRY_TEXTPROTO)
    suite = generate_matrices._parse_registry(temp_file.full_path)

    self.assertIsNotNone(suite)
    self.assertIsInstance(suite, benchmark_config_pb2.BenchmarkSuite)
    self.assertLen(suite.configs, 2)

    # Check first config details
    config1 = suite.configs[0]
    self.assertEqual(config1.name, "gemma3_1b_flax_call")
    self.assertEqual(config1.hardware_category, benchmark_config_pb2.GPU_B200)
    self.assertEqual(config1.topology.num_hosts, 1)
    self.assertEqual(config1.topology.num_devices_per_host, 1)
    self.assertFalse(config1.topology.multi_host)  # Default value
    self.assertFalse(config1.topology.multi_device)  # Default value
    self.assertIn(benchmark_config_pb2.POSTSUBMIT, config1.run_frequencies)
    self.assertEqual(
        config1.hlo_gcs_bucket_path,
        "https://storage.googleapis.com/xla-benchmarking-temp/gemma3_1b_flax_call.hlo",
    )
    self.assertEqual(config1.hlo_path, "")
    self.assertIn("--num_repeat=5", config1.runtime_flags)
    self.assertIn("blocking_presubmit_test", config1.github_labels)

    # Check second config details
    config2 = suite.configs[1]
    self.assertEqual(config2.name, "gemma2_2b_keras_jax")
    self.assertEqual(config2.hardware_category, benchmark_config_pb2.CPU_X86)
    self.assertIn(benchmark_config_pb2.PRESUBMIT, config2.run_frequencies)
    self.assertIn(benchmark_config_pb2.POSTSUBMIT, config2.run_frequencies)
    self.assertEqual(config2.hlo_path, "benchmarks/hlo/gemma2_2b_keras_jax.hlo")
    self.assertEqual(config2.hlo_gcs_bucket_path, "")
    self.assertIn(benchmark_config_pb2.CPU_TIME, config2.target_metrics)
    self.assertIn(benchmark_config_pb2.PEAK_CPU_MEMORY, config2.target_metrics)

  def test_parse_registry_file_not_found(self):
    """Tests parsing a non-existent file."""
    suite = generate_matrices._parse_registry(
        "/tmp/non_existent_file.textproto"
    )
    self.assertIsNone(suite)

  def test_parse_registry_parse_error(self):
    """Tests parsing a malformed TextProto file."""
    malformed_content = "configs { name: unterminated_string"
    temp_file = self.create_tempfile(content=malformed_content)
    suite = generate_matrices._parse_registry(temp_file.full_path)
    self.assertIsNone(suite)

  @parameterized.named_parameters(
      (
          "gemma3_1b_flax_call_b200_1h_1d",
          benchmark_config_pb2.GPU_B200,
          1,
          1,
          "linux-x86-a4-224-b200-1gpu",
          "us-central1-docker.pkg.dev/tensorflow-sigs/tensorflow/ml-build-cuda12.8-cudnn9.8:latest",
      ),
      (
          "gemma2_2b_keras_jax_x86_1h_1d",
          benchmark_config_pb2.CPU_X86,
          1,
          1,
          "linux-x86-n2-128",
          "us-central1-docker.pkg.dev/tensorflow-sigs/tensorflow/ml-build:latest",
      ),
      (
          "gemma2_2b_keras_jax_arm64_1h_1d",
          benchmark_config_pb2.CPU_ARM64,
          1,
          1,
          "linux-arm64-c4a-64",
          "us-central1-docker.pkg.dev/tensorflow-sigs/tensorflow/ml-build-arm64:latest",
      ),
  )
  def test_get_runner_info(
      self, hw, hosts, devs, expected_label, expected_image
  ):
    """Tests the mapping from config to runner label and container image."""
    config = benchmark_config_pb2.BenchmarkConfig(
        name="test_config",
        hardware_category=hw,
        topology=benchmark_config_pb2.ExecutionTopology(
            num_hosts=hosts, num_devices_per_host=devs
        ),
    )
    label, image = generate_matrices.get_runner_info(config)
    self.assertEqual(label, expected_label)
    self.assertEqual(image, expected_image)

  def test_generate_matrix_from_default_registry(self):
    """Tests generating the full matrix from the default registry data."""
    suite = benchmark_config_pb2.BenchmarkSuite()
    text_format.Parse(DEFAULT_REGISTRY_TEXTPROTO, suite)

    matrix = generate_matrices.generate_matrix(suite)

    self.assertIn("include", matrix)
    include_list = matrix["include"]

    # Expected number of entries:
    # config1 (gemma3): 1 run_frequency (POSTSUBMIT) -> 1 entry
    # config2 (gemma2): 2 run_frequencies (PRESUBMIT, POSTSUBMIT) -> 2 entries
    self.assertLen(include_list, 3)

    # --- Verification for gemma3_1b_flax_call (POSTSUBMIT) ---
    entry1_iter = (
        e
        for e in include_list
        if e["benchmark_name"] == "gemma3_1b_flax_call"
        and e["run_frequency"] == "POSTSUBMIT"  # Filter explicitly
    )
    # Expect only one entry for this combination of benchmark and run frequency.
    entry1 = more_itertools.one(entry1_iter)
    self.assertEqual(entry1["config_id"], "gemma3_1b_flax_call_b200_1h_1d")
    self.assertEqual(entry1["run_frequency"], "POSTSUBMIT")
    self.assertEqual(entry1["runner_label"], "linux-x86-a4-224-b200-1gpu")
    self.assertEqual(
        entry1["container_image"],
        "us-central1-docker.pkg.dev/tensorflow-sigs/tensorflow/ml-build-cuda12.8-cudnn9.8:latest",
    )
    self.assertEqual(
        entry1["hlo_location"],
        "https://storage.googleapis.com/xla-benchmarking-temp/gemma3_1b_flax_call.hlo",
    )
    self.assertTrue(entry1["is_gcs_hlo"])
    # Note: Target metrics are mapped by hardware category.
    self.assertListEqual(
        entry1["target_metrics"],
        ["GPU_DEVICE_TIME", "GPU_DEVICE_MEMCPY_TIME"],
    )
    self.assertEqual(entry1["xla_compilation_flags"], "[]")
    self.assertEqual(entry1["runtime_flags"], json.dumps(["--num_repeat=5"]))
    # Note: Target metrics are mapped by hardware category.
    self.assertEqual(entry1["required_hardware_category"], "GPU_B200")
    expected_topology1 = {
        "multi_host": False,
        "multi_device": False,
        "num_hosts": 1,
        "num_devices_per_host": 1,
    }
    self.assertEqual(entry1["topology"], json.dumps(expected_topology1))
    self.assertEqual(
        entry1["github_labels"], json.dumps(["blocking_presubmit_test"])
    )

    # --- Verification for gemma2_2b_keras_jax (PRESUBMIT) ---
    entry2_iter = (
        e
        for e in include_list
        if e["benchmark_name"] == "gemma2_2b_keras_jax"
        and e["run_frequency"] == "PRESUBMIT"
    )
    # Expect only one entry for this combination of benchmark and run frequency.
    entry2 = more_itertools.one(entry2_iter)
    self.assertEqual(entry2["config_id"], "gemma2_2b_keras_jax_x86_1h_1d")
    self.assertEqual(entry2["run_frequency"], "PRESUBMIT")
    self.assertEqual(entry2["runner_label"], "linux-x86-n2-128")
    self.assertEqual(
        entry2["container_image"],
        "us-central1-docker.pkg.dev/tensorflow-sigs/tensorflow/ml-build:latest",
    )
    self.assertEqual(
        entry2["hlo_location"], "benchmarks/hlo/gemma2_2b_keras_jax.hlo"
    )
    self.assertFalse(entry2["is_gcs_hlo"])
    # Note: Target metrics are mapped by hardware category.
    self.assertListEqual(entry2["target_metrics"], ["CPU_TIME"])
    self.assertEqual(entry2["xla_compilation_flags"], "[]")
    self.assertEqual(entry2["runtime_flags"], json.dumps(["--num_repeat=5"]))
    self.assertEqual(entry2["required_hardware_category"], "CPU_X86")
    expected_topology2 = {
        "multi_host": False,
        "multi_device": False,
        "num_hosts": 1,
        "num_devices_per_host": 1,
    }
    self.assertEqual(entry2["topology"], json.dumps(expected_topology2))
    self.assertEqual(
        entry2["github_labels"], json.dumps(["blocking_presubmit_test"])
    )

    # --- Verification for gemma2_2b_keras_jax (POSTSUBMIT) ---
    entry3_iter = (
        e
        for e in include_list
        if e["benchmark_name"] == "gemma2_2b_keras_jax"
        and e["run_frequency"] == "POSTSUBMIT"
    )
    # Expect only one entry for this combination of benchmark and run frequency.
    entry3 = more_itertools.one(entry3_iter)
    self.assertEqual(entry3["config_id"], "gemma2_2b_keras_jax_x86_1h_1d")
    self.assertEqual(entry3["run_frequency"], "POSTSUBMIT")
    self.assertEqual(entry3["runner_label"], "linux-x86-n2-128")
    self.assertListEqual(entry3["target_metrics"], ["CPU_TIME"])


if __name__ == "__main__":
  absltest.main()

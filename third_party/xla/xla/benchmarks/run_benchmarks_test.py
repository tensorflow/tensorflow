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

"""Unit tests for run_benchmarks script."""

import dataclasses
import os
from unittest import mock

from absl.testing import absltest
from absl.testing import flagsaver
from jax.experimental.pallas import tpu as pltpu
import pandas as pd

from xla.benchmarks import benchmark_configs
from xla.benchmarks import run_benchmarks_lib
from xla.benchmarks.core import benchmark
from xla.benchmarks.jax_microbenchmarks import jax_profiler_utils


class FakeBenchmark(benchmark.Benchmark):

  def get_input_shapes_and_dtypes(self):
    return []

  def target_fn(self):
    return lambda: None

  def kernel_name(self):
    return "fake"

  def run(self, **kwargs):
    return [
        jax_profiler_utils.JaxProfilerResult(runtimes_us=[100.0], flops=1e9)
    ]


@dataclasses.dataclass(frozen=True)
class FakeBenchmarkConfig(benchmark.BenchmarkConfig):
  m: int = 128

  def get_benchmark(self) -> benchmark.Benchmark:
    return FakeBenchmark()


class RunBenchmarksTest(absltest.TestCase):

  def test_run_benchmark_suite(self):
    fake_config = FakeBenchmarkConfig()

    with mock.patch.object(
        benchmark_configs,
        "BENCHMARK_FACTORIES",
        {"test_bm": lambda chip_version=None: [fake_config]},
    ), mock.patch.object(
        pltpu,
        "get_tpu_info",
        return_value=mock.MagicMock(chip_version=pltpu.ChipVersion.TPU_V5E),
    ):
      df = run_benchmarks_lib.run_benchmark_suite("test_bm", repeat=1, runs=1)
      self.assertIsInstance(df, pd.DataFrame)
      self.assertLen(df, 1)
      self.assertEqual(df["m"].iloc[0], 128)
      self.assertEqual(df["latency_us"].iloc[0], 100.0)
      self.assertEqual(df["flops"].iloc[0], 1e9)

  def test_unknown_benchmark_suite_raises(self):
    with self.assertRaises(KeyError):
      run_benchmarks_lib.run_benchmark_suite("non_existent_suite")

  def test_main_with_flags(self):
    fake_config = FakeBenchmarkConfig()
    temp_dir = self.create_tempdir()
    csv_file = os.path.join(temp_dir.full_path, "results.csv")

    with mock.patch.object(
        benchmark_configs,
        "BENCHMARK_FACTORIES",
        {"dense_matmul": lambda chip_version=None: [fake_config]},
    ), mock.patch.object(
        pltpu,
        "get_tpu_info",
        return_value=mock.MagicMock(chip_version=pltpu.ChipVersion.TPU_V5E),
    ), flagsaver.flagsaver(
        benchmarks=["dense_matmul"],
        csv_path=csv_file,
    ):
      run_benchmarks_lib.main()
      self.assertIsNotNone(pd.read_csv(csv_file))


if __name__ == "__main__":
  absltest.main()

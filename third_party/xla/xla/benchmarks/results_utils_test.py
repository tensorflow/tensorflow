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

"""Unit tests for results_utils."""

import os

from absl.testing import absltest
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import pandas as pd

from xla.benchmarks import results_utils
from xla.benchmarks.jax_microbenchmarks import jax_profiler_utils
from xla.benchmarks.pallas_microbenchmarks import dense_matmul_lib


class ResultsUtilsTest(absltest.TestCase):

  def test_extract_metrics(self):
    prof_res1 = jax_profiler_utils.JaxProfilerResult(
        runtimes_us=[100.0, 200.0], flops=1e12
    )
    prof_res2 = jax_profiler_utils.JaxProfilerResult(
        runtimes_us=[300.0], flops=1e12
    )
    metrics = results_utils.extract_metrics([prof_res1, prof_res2])
    self.assertAlmostEqual(metrics["latency_us"], 200.0)
    self.assertEqual(metrics["flops"], 1e12)

    # Test empty / None profiler results
    empty_metrics = results_utils.extract_metrics([None])
    self.assertIsNone(empty_metrics["latency_us"])
    self.assertIsNone(empty_metrics["flops"])

  def test_create_results_table(self):
    cfg = dense_matmul_lib.DenseMatmulConfig(
        m=1024,
        k=1024,
        n=1024,
        block_m=128,
        block_k=128,
        block_n=128,
        lhs_mem=pltpu.HBM,
        rhs_mem=pltpu.HBM,
        out_mem=pltpu.HBM,
        lhs_dtype=jnp.bfloat16,
        rhs_dtype=jnp.bfloat16,
        out_dtype=jnp.float32,
        acc_dtype=jnp.float32,
    )
    prof_res = jax_profiler_utils.JaxProfilerResult(
        runtimes_us=[150.0], flops=5e11
    )
    df = results_utils.create_results_table([(cfg, [prof_res])])
    self.assertIsInstance(df, pd.DataFrame)
    self.assertLen(df, 1)
    self.assertIn("m", df.columns)
    self.assertIn("latency_us", df.columns)
    self.assertIn("flops", df.columns)
    self.assertEqual(df["m"].iloc[0], 1024)
    self.assertEqual(df["latency_us"].iloc[0], 150.0)
    self.assertEqual(df["flops"].iloc[0], 5e11)

  def test_write_and_save_csv(self):
    temp_dir = self.create_tempdir()
    df1 = pd.DataFrame([{"a": 1, "b": 2}])
    df2 = pd.DataFrame([{"c": 3, "d": 4}])

    # Test single benchmark CSV write
    csv_single = os.path.join(temp_dir.full_path, "single.csv")
    results_utils.save_all_results_to_csv({"bm1": df1}, csv_single)
    self.assertTrue(os.path.exists(csv_single))
    read_df = pd.read_csv(csv_single)
    self.assertEqual(read_df.to_dict(orient="records"), [{"a": 1, "b": 2}])

    # Test multi-benchmark CSV write to directory
    dir_path = os.path.join(temp_dir.full_path, "csv_folder")
    os.makedirs(dir_path, exist_ok=True)
    results_utils.save_all_results_to_csv({"bm1": df1, "bm2": df2}, dir_path)
    self.assertTrue(os.path.exists(os.path.join(dir_path, "bm1.csv")))
    self.assertTrue(os.path.exists(os.path.join(dir_path, "bm2.csv")))


if __name__ == "__main__":
  absltest.main()

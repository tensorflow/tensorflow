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

"""Unit tests for benchmark_configs."""

import unittest.mock

from absl.testing import absltest
from absl.testing import parameterized
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

from xla.benchmarks import benchmark_configs
from xla.benchmarks import results_utils
from xla.benchmarks.dma_microbenchmarks import memory_base
from xla.benchmarks.jax_microbenchmarks import matmul_lib
from xla.benchmarks.pallas_microbenchmarks import dense_matmul_lib
from xla.benchmarks.pallas_microbenchmarks import subchannel_matmul_lib


class BenchmarkConfigsTest(parameterized.TestCase):

  def test_dense_matmul_configs(self):
    chip = pltpu.ChipVersion.TPU_V5E
    configs = benchmark_configs.get_dense_matmul_configs(chip_version=chip)
    # Expected total configs:
    # M in [1024, 2048] ->
    #   2 sizes * 5 dtype pairs * 2 out_dtypes * 2 mems (HBM, VMEM) = 40
    # M in [4096, 8192, 16384, 32768] ->
    #   4 sizes * 4 dtype pairs * 2 out_dtypes * 1 mem (HBM) = 40
    # Total = 32 configs.
    self.assertLen(configs, 80)

    for cfg in configs:
      self.assertIsInstance(cfg, dense_matmul_lib.DenseMatmulConfig)
      self.assertEqual(cfg.m, cfg.k)
      self.assertEqual(cfg.m, cfg.n)
      self.assertIn(cfg.m, [1024, 2048, 4096, 8192, 16384, 32768])
      self.assertIn(
          (cfg.lhs_dtype, cfg.rhs_dtype),
          [
              (jnp.bfloat16, jnp.bfloat16),
              (jnp.bfloat16, jnp.float8_e4m3fn),
              (jnp.bfloat16, jnp.int4),
              (jnp.float8_e4m3fn, jnp.float8_e4m3fn),
              (jnp.float8_e4m3fn, jnp.int4),
          ],
      )
      self.assertIn(cfg.out_dtype, [jnp.float32, jnp.bfloat16])
      self.assertEqual(cfg.acc_dtype, jnp.float32)
      self.assertEqual(cfg.lhs_mem, cfg.rhs_mem)
      self.assertEqual(cfg.lhs_mem, cfg.out_mem)

      if cfg.m in (1024, 2048):
        self.assertIn(cfg.lhs_mem, [pltpu.HBM, pltpu.VMEM])
      else:
        self.assertEqual(cfg.lhs_mem, pltpu.HBM)

      self.assertGreater(cfg.block_m, 0)
      self.assertGreater(cfg.block_k, 0)
      self.assertGreater(cfg.block_n, 0)

  def test_subchannel_matmul_configs(self):
    chip = pltpu.ChipVersion.TPU_V5E
    configs = benchmark_configs.get_subchannel_matmul_configs(chip_version=chip)
    # Expected: 2 configs (HBM and VMEM)
    self.assertLen(configs, 2)

    mems = set()
    for cfg in configs:
      self.assertIsInstance(cfg, subchannel_matmul_lib.SubchannelMatmulConfig)
      self.assertEqual(cfg.m, 128)
      self.assertEqual(cfg.k, 8192)
      self.assertEqual(cfg.n, 4096)
      self.assertEqual(cfg.lhs_dtype, jnp.bfloat16)
      self.assertEqual(cfg.rhs_dtype, jnp.bfloat16)
      self.assertEqual(cfg.out_dtype, jnp.bfloat16)
      self.assertEqual(cfg.subchannel_size, 1024)
      self.assertEqual(cfg.lhs_quantized_dtype, jnp.float8_e4m3fn)
      self.assertEqual(cfg.rhs_quantized_dtype, jnp.int4)
      self.assertFalse(cfg.pre_quantize_lhs)
      self.assertEqual(cfg.lhs_mem, cfg.rhs_mem)
      self.assertEqual(cfg.lhs_mem, cfg.out_mem)
      self.assertGreater(cfg.block_m, 0)
      self.assertGreater(cfg.block_k, 0)
      self.assertGreater(cfg.block_n, 0)
      mems.add(cfg.lhs_mem)

    self.assertEqual(mems, {pltpu.HBM, pltpu.VMEM})

  def test_jax_matmul_configs(self):
    configs = benchmark_configs.get_jax_matmul_configs()
    # 6 dim sizes * 5 dtype pairs * 2 out dtypes = 60 configs.
    self.assertLen(configs, 60)
    for cfg in configs:
      self.assertIsInstance(cfg, matmul_lib.JaxMatmulConfig)
      self.assertEqual(cfg.b, 1)
      self.assertEqual(cfg.m, cfg.k)
      self.assertEqual(cfg.m, cfg.n)
      self.assertIn(cfg.m, [1024, 2048, 4096, 8192, 16384, 32768])
      self.assertIn(
          (cfg.lhs_dtype, cfg.rhs_dtype),
          [
              (jnp.bfloat16, jnp.bfloat16),
              (jnp.bfloat16, jnp.float8_e4m3fn),
              (jnp.bfloat16, jnp.int4),
              (jnp.float8_e4m3fn, jnp.float8_e4m3fn),
              (jnp.float8_e4m3fn, jnp.int4),
          ],
      )
      self.assertIn(cfg.out_dtype, [jnp.float32, jnp.bfloat16])

  def test_dma_configs(self):
    configs = benchmark_configs.get_dma_configs()
    self.assertLen(configs, 17)
    counts_by_suite = {}
    for cfg in configs:
      self.assertIsInstance(cfg, memory_base.DmaBenchmarkConfig)
      self.assertStartsWith(cfg.test_name, "test_")
      self.assertEqual(
          cfg.as_dict(), {"suite": cfg.suite, "test": cfg.test_name}
      )
      counts_by_suite[cfg.suite] = counts_by_suite.get(cfg.suite, 0) + 1
    self.assertEqual(
        counts_by_suite,
        {
            "local_dma": 12,
            "host_dma": 2,
            "chip_to_chip_dma": 1,
            "chiplet_to_chiplet_dma": 2,
        },
    )

  def test_dma_benchmark_end_to_end_and_skip(self):
    # 1. On CPU without TPU, DmaBenchmark.run() catches SkipTest -> [None].
    cfg_skip = benchmark_configs.get_dma_configs()[0]
    self.assertEqual(cfg_skip.get_benchmark().run(), [None])

    # 2. With TPU info patched using spec_set=True, run() executes the test
    # method, records bandwidth/latency metrics, and populates the table.
    class _FakeDmaSuite(memory_base.MemoryBenchmarks):

      def setUp(self):
        absltest.TestCase.setUp(self)
        self.latencies_us = []
        self.metrics = {}

      def test_bw(self):
        self._print_bandwidth_statistics(
            dma_size_kib=1024,
            num_dmas=4,
            latencies_us=[10.0, 20.0],
        )

    cfg = memory_base.DmaBenchmarkConfig(
        suite="local_dma", test_class=_FakeDmaSuite, test_name="test_bw"
    )
    with unittest.mock.patch.object(
        pltpu,
        "get_tpu_info",
        spec_set=True,
        return_value=pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_V5E, 1),
    ):
      prof_results = cfg.get_benchmark().run()

    df = results_utils.create_results_table([(cfg, prof_results)])
    self.assertEqual(
        list(df.columns),
        [
            "suite",
            "test",
            "dma_size_kib",
            "num_dmas",
            "bandwidth_gbps",
            "latency_us",
            "flops",
        ],
    )
    self.assertEqual(df["suite"].iloc[0], "local_dma")
    self.assertEqual(df["test"].iloc[0], "test_bw")
    self.assertEqual(df["dma_size_kib"].iloc[0], 1024)
    self.assertEqual(df["num_dmas"].iloc[0], 4)
    self.assertAlmostEqual(df["latency_us"].iloc[0], 15.0)
    self.assertGreater(df["bandwidth_gbps"].iloc[0], 0.0)


if __name__ == "__main__":
  absltest.main()

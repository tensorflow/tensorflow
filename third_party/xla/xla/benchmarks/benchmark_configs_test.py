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

from absl.testing import absltest
from absl.testing import parameterized
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

from xla.benchmarks import benchmark_configs
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


if __name__ == "__main__":
  absltest.main()

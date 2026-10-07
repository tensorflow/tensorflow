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

"""Unit tests for JAX profiler utilities, verifying DataFrame generation."""

from absl.testing import absltest
import jax
import jax.numpy as jnp

from xla.benchmarks.jax_microbenchmarks import jax_profiler_utils


class JaxProfilerUtilsTest(absltest.TestCase):

  def _run_test(self, use_profiler: bool):
    kernel_name = "test_matmul"

    def matmul_fn(x, y):
      return jnp.matmul(x, y)

    matmul_jit = jax.jit(matmul_fn)

    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (500, 500))
    y = jax.random.normal(key, (500, 500))

    # Warmup
    matmul_jit(x, y).block_until_ready()

    with jax_profiler_utils.JaxProfiler(
        kernel_name, enable_profiling=use_profiler
    ) as profiler:
      with jax.profiler.TraceAnnotation(kernel_name):
        matmul_jit(x, y).block_until_ready()

    return profiler

  def test_profiler_returns_dataframe(self):
    profiler = self._run_test(use_profiler=True)
    self.assertIsNotNone(profiler.result)
    df = profiler.result.as_dataframe()
    self.assertIsNotNone(df)
    self.assertIn("runtime_us", df.columns)
    self.assertIn("flops", df.columns)
    self.assertNotEmpty(df)

  def test_profiler_with_profiling_disabled(self):
    profiler = self._run_test(use_profiler=False)
    self.assertIsNotNone(profiler.result)
    self.assertEmpty(profiler.result.runtimes_us)
    self.assertEqual(profiler.result.flops, 0.0)


if __name__ == "__main__":
  absltest.main()

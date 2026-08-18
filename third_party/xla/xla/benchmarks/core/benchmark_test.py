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

"""Unit tests for the JAX microbenchmark core base class."""

import types
from typing import Any, Callable, Self, Sequence
from unittest import mock

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np

from xla.benchmarks.core import benchmark  # pylint: disable=g-direct-tensorflow-import
from xla.benchmarks.jax_microbenchmarks import jax_profiler_utils  # pylint: disable=g-direct-tensorflow-import

InputSpec = benchmark.InputSpec


class FakeJaxProfiler:
  """Fake JaxProfiler context manager for unit testing."""

  def __init__(self, kernel_name: str) -> None:
    self.kernel_name = kernel_name
    self.result = None

  def __enter__(self) -> Self:
    self._count = 0
    self._orig_block = benchmark._block_until_ready

    def _mock_block(tree):
      self._count += 1
      return self._orig_block(tree)

    self._patcher = mock.patch.object(
        benchmark, "_block_until_ready", side_effect=_mock_block
    )
    self._patcher.start()
    return self

  def __exit__(
      self,
      exc_type: type[BaseException] | None,
      exc_val: BaseException | None,
      exc_tb: types.TracebackType | None,
  ) -> None:
    """Stops the patcher and sets the result.

    Args:
      exc_type: Exception type.
      exc_val: Exception value.
      exc_tb: Exception traceback.
    """
    self._patcher.stop()
    self.result = jax_profiler_utils.JaxProfilerResult(
        runtimes_us=np.ones(self._count, dtype=np.float64),  # pyrefly: ignore[bad-argument-type]
        flops=0.0,
    )


class SimpleBenchmark(benchmark.Benchmark):
  """Simple benchmark subclass for testing."""

  def get_input_shapes_and_dtypes(self) -> Sequence[InputSpec]:
    return [
        InputSpec(shape=(4, 4), dtype=jnp.float32),
        InputSpec(shape=(4, 4), dtype=jnp.float32),
    ]

  def target_fn(self) -> Callable[..., Any]:

    @jax.jit
    def _target_fn(x, y):
      return jnp.add(x, y)

    return _target_fn

  def kernel_name(self) -> str:
    return "simple_add"

  def reference_fn(self) -> Callable[..., Any] | None:

    def _reference_fn(x, y):
      return x + y

    return _reference_fn


class PyTreeBenchmark(benchmark.Benchmark):
  """Benchmark returning PyTree tuple outputs."""

  def get_input_shapes_and_dtypes(self) -> Sequence[InputSpec]:
    return [
        InputSpec(shape=(2, 4), dtype=jnp.float32),
        InputSpec(shape=(2, 4), dtype=jnp.int32),
        InputSpec(shape=(2, 4), dtype=jnp.bool_),
    ]

  def target_fn(self) -> Callable[..., Any]:

    @jax.jit
    def _target_fn(x, y, b):
      return (x * 2.0, (y + 1, jnp.logical_not(b)))

    return _target_fn

  def kernel_name(self) -> str:
    return "pytree_outputs"

  def reference_fn(self) -> Callable[..., Any] | None:

    def _reference_fn(x, y, b):
      return (x * 2.0, (y + 1, jnp.logical_not(b)))

    return _reference_fn


class BenchmarkTest(absltest.TestCase):

  def setUp(self) -> None:
    super().setUp()
    self.enter_context(
        mock.patch.object(jax_profiler_utils, "JaxProfiler", FakeJaxProfiler)
    )

  def test_generate_inputs_zeros(self):
    bm = SimpleBenchmark()
    inputs = bm.generate_inputs(use_random_data=False)
    self.assertLen(inputs, 2)
    self.assertIsNotNone(inputs[0])
    self.assertEqual(inputs[0].shape, (4, 4))
    self.assertEqual(inputs[0].dtype, jnp.float32)
    np.testing.assert_array_equal(inputs[0], 0)

  def test_generate_inputs_random(self):
    bm = SimpleBenchmark()
    inputs = bm.generate_inputs(use_random_data=True)
    self.assertLen(inputs, 2)
    self.assertIsNotNone(inputs[0])
    self.assertEqual(inputs[0].shape, (4, 4))
    self.assertEqual(inputs[0].dtype, jnp.float32)
    self.assertFalse(np.array_equal(inputs[0], 0))

  def test_run_benchmark(self):
    bm = SimpleBenchmark()
    results = bm.run(
        repeat=3,
        runs=2,
        use_random_data=True,
        check_numerics=True,
    )
    self.assertLen(results, 2)
    for result in results:
      self.assertIsInstance(result, jax_profiler_utils.JaxProfilerResult)
      self.assertLen(result.runtimes_us, 3)

  def test_pytree_outputs(self):
    bm = PyTreeBenchmark()
    inputs = bm.generate_inputs(use_random_data=True)
    self.assertLen(inputs, 3)
    self.assertIsNotNone(inputs[0])
    self.assertIsNotNone(inputs[1])
    self.assertIsNotNone(inputs[2])
    self.assertEqual(inputs[0].dtype, jnp.float32)
    self.assertEqual(inputs[1].dtype, jnp.int32)
    self.assertEqual(inputs[2].dtype, jnp.bool_)

    results = bm.run(
        repeat=2,
        runs=1,
        use_random_data=True,
        check_numerics=True,
    )
    self.assertLen(results, 1)
    result = results[0]
    self.assertIsInstance(result, jax_profiler_utils.JaxProfilerResult)
    self.assertLen(result.runtimes_us, 2)

  def test_invalid_repeat_or_runs_raises(self):
    bm = SimpleBenchmark()
    with self.assertRaises(ValueError):
      bm.run(repeat=0)
    with self.assertRaises(ValueError):
      bm.run(runs=0)


if __name__ == "__main__":
  absltest.main()

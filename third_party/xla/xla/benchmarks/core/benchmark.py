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

"""Core benchmark base classes for JAX microbenchmarks."""

import abc
from collections.abc import Callable, Sequence
import dataclasses
from typing import Any

from absl import logging
import jax
import jax.numpy as jnp
import numpy as np

from xla.benchmarks.jax_microbenchmarks import jax_profiler_utils  # pylint: disable=g-direct-tensorflow-import


@dataclasses.dataclass(frozen=True)
class InputSpec:
  """Specification for input tensors to a benchmark.

  Attributes:
    shape: Shape of the input tensor.
    dtype: Data type of the input tensor.
  """
  shape: Sequence[int]
  dtype: np.dtype


def _block_until_ready(tree: Any) -> Any:
  """Recursively calls block_until_ready on leaves if supported."""
  return jax.tree_util.tree_map(
      lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
      tree,
  )


class Benchmark(abc.ABC):
  """Base class for JAX microbenchmarks.

  Subclasses define:
    - `get_input_shapes_and_dtypes()`: shapes and dtypes of input tensors.
    - `target_fn()`: target function to benchmark.
    - `kernel_name()`: name of the HLO of interest for profiling.
    - `reference_fn()`: (optional) reference implementation for verifying
      numerical accuracy.
  """

  @abc.abstractmethod
  def get_input_shapes_and_dtypes(self) -> Sequence[InputSpec | None]:
    """Returns shape and dtype specifications for input tensors."""

  @abc.abstractmethod
  def target_fn(self) -> Callable[..., Any]:
    """Returns the target function to benchmark."""

  @abc.abstractmethod
  def kernel_name(self) -> str:
    """Returns the name of the HLO being benchmarked."""

  def reference_fn(self) -> Callable[..., Any] | None:
    """Optional reference implementation for numerical verification."""
    return None

  def generate_inputs(
      self, use_random_data: bool = True, seed: int = 1234
  ) -> list[np.ndarray | None]:
    """Instantiate input tensors and wait for them to be ready.

    Args:
      use_random_data: Whether to use random data for input tensors, or zeros.
      seed: Random seed for reproducibility.

    Returns:
      A list of input tensors.
    """
    if use_random_data:
      rng = np.random.default_rng(seed)

      def _generate(dtype, shape):
        if jnp.issubdtype(dtype, jnp.floating):
          arr = rng.normal(size=shape)
        elif jnp.issubdtype(dtype, jnp.integer):
          arr = rng.integers(
              jnp.iinfo(dtype).min, jnp.iinfo(dtype).max, size=shape
          )
        elif jnp.issubdtype(dtype, jnp.bool_):
          arr = rng.integers(0, 2, size=shape)
        else:
          raise ValueError(f"Unsupported dtype: {dtype}")
        return arr.astype(dtype)

      return [
          _generate(spec.dtype, spec.shape) if spec is not None else None
          for spec in self.get_input_shapes_and_dtypes()
      ]
    else:
      return [
          np.zeros(spec.shape, dtype=spec.dtype) if spec is not None else None
          for spec in self.get_input_shapes_and_dtypes()
      ]

  def run(
      self,
      *,
      repeat: int = 1,
      runs: int = 1,
      use_random_data: bool = True,
      check_numerics: bool = False,
      rtol: float = 1e-2,
      atol: float = 1e-2,
  ) -> list[jax_profiler_utils.JaxProfilerResult | None]:
    """Runs the benchmark.

    Args:
      repeat: Number of times target_fn is executed per benchmark run. The same
        inputs are used for each repeat.
      runs: Number of end-to-end benchmark iterations. A different set of inputs
        are generated for each run.
      use_random_data: Whether to use random data for input tensors, or zeros.
      check_numerics: Whether to verify output accuracy against reference_fn.
        This check is only performed on the output of the last repeat, but for
        all runs.
      rtol: Relative tolerance for numerical verification.
      atol: Absolute tolerance for numerical verification.

    Returns:
      A list of JaxProfilerResult objects, one for each run.
    """
    if repeat < 1:
      raise ValueError("repeat must be at least 1.")
    if runs < 1:
      raise ValueError("runs must be at least 1.")

    kernel_name = self.kernel_name()
    logging.info("xprof_hlos_of_interest=[%s.1]", kernel_name)

    profiler_results = []
    for _ in range(runs):
      inputs = self.generate_inputs(use_random_data=use_random_data)

      with jax_profiler_utils.JaxProfiler(kernel_name) as profiler:
        target_fn = self.target_fn()
        result = target_fn(*inputs)
        _block_until_ready(result)

        for _ in range(repeat - 1):
          result = target_fn(*inputs)
          _block_until_ready(result)

      if profiler.result is not None:
        logging.info(
            "Profiler results:\n%s", profiler.result.as_dataframe().to_string()
        )
      else:
        logging.warning("No profiler results found.")

      if check_numerics:
        reference_fn = self.reference_fn()
        if reference_fn is not None:
          ref = reference_fn(*inputs)
          _block_until_ready(ref)
          jax.tree_util.tree_map(
              lambda r, e: np.testing.assert_allclose(
                  r, e, rtol=rtol, atol=atol
              ),
              result,
              ref,
          )
        else:
          logging.warning(
              "check_numerics requested but reference_fn returned None."
          )

      profiler_results.append(profiler.result)

    return profiler_results

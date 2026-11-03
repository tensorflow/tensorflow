# Copyright 2026 The OpenXLA Authors.
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

"""Pure JAX reference implementations for tgmm."""

import jax
import jax.numpy as jnp

DRHS_DIM_NUMS = jax.lax.RaggedDotDimensionNumbers(
    dot_dimension_numbers=(([0], [0]), ([], [])),
    lhs_ragged_dimensions=[0],
    rhs_group_dimensions=[],
)


def tgmm_reference(
    lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array
) -> jax.Array:
  """Reference backward ragged_dot (tgmm) in JAX."""
  return jax.lax.ragged_dot_general(
      lhs,
      rhs,
      group_sizes,
      ragged_dot_dimension_numbers=DRHS_DIM_NUMS,
      preferred_element_type=jnp.float32,
  ).astype(lhs.dtype)


tgmm_ref = tgmm_reference

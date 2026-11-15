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

"""Pure JAX reference implementations for gmm."""

from typing import Any
import jax
import jax.numpy as jnp


def gmm_reference(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    *,
    return_residuals: bool = False
) -> Any:
  """Reference forward ragged_dot in JAX with zeroed rows past sum(group_sizes)."""
  raw_out = jax.lax.ragged_dot(
      lhs, rhs, group_sizes, preferred_element_type=jnp.float32
  ).astype(lhs.dtype)
  covered = jnp.sum(jnp.maximum(group_sizes.astype(jnp.int32), 0))
  row_idx = jax.lax.broadcasted_iota(jnp.int32, raw_out.shape, 0)
  out = jnp.where(row_idx < covered, raw_out, jnp.zeros_like(raw_out))
  return (out, out) if return_residuals else out


gmm_ref = gmm_reference

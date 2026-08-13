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

"""Utility functions for memory operations used in XLA benchmarks.
"""

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


@jax.jit
def copy_kernel(
    x_hbm_ref: jax.Array, x_vmem_ref: jax.Array, copy_sem: jax.Array
) -> None:
  """Kernel for copying an HBM array to VMEM.

  Args:
    x_hbm_ref: The HBM array reference to copy from.
    x_vmem_ref: The VMEM array reference to copy to.
    copy_sem: The semaphore to use for the copy.
  """
  pltpu.async_copy(x_hbm_ref, x_vmem_ref, copy_sem).wait()


def copy_to_vmem(arr: jax.Array) -> jax.Array:
  """Utility function to copy an HBM array to VMEM.

  Used for benchmarks where one or more operands are in VMEM.

  Args:
    arr: The HBM array to copy to VMEM.

  Returns:
    The array in VMEM.
  """
  return pl.pallas_call(
      copy_kernel,
      out_shape=pltpu.VMEM(arr.shape, arr.dtype),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          in_specs=[
              pl.BlockSpec(memory_space=pltpu.HBM),
          ],
          out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
          grid=(1,),
          scratch_shapes=[
              pltpu.SemaphoreType.DMA(()),
          ],
      ),
  )(arr)

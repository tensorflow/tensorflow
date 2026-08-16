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

"""JAX microbenchmarks for measuring local DMA bandwidth and latency."""

from absl import flags
from absl.testing import absltest
import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
from xla.benchmarks.dma_microbenchmarks import memory_base  # pylint: disable=g-direct-tensorflow-import


_VMEM_DMA_SIZE_KIB = flags.DEFINE_integer(
    "vmem_dma_size_kib",
    default=None,
    help="Size of a single VMEM DMA in KiB. Default: VMEM capacity - 1 MiB.",
)
_SMEM_DMA_SIZE_KIB = flags.DEFINE_integer(
    "smem_dma_size_kib",
    default=None,
    help="Size of a single SMEM DMA in KiB. Default: SMEM capacity - 16 KiB.",
)
_NUMBER_OF_DMAS = flags.DEFINE_integer(
    "number_of_dmas",
    default=16,
    help="Number of overlapping DMAs running in parallel. Default: 16",
)
_NUMBER_OF_MEASUREMENTS = flags.DEFINE_integer(
    "number_of_measurements",
    default=5,
    help="Number of measurements to take. Default: 5",
)


class LocalDmaBenchmarks(memory_base.MemoryBenchmarks):
  """Test suite measuring local DMA bandwidth and latency."""

  def setUp(self):
    super().setUp()
    tpu_info = pltpu.get_tpu_info()
    vmem_capacity_kib = tpu_info.vmem_capacity_bytes // 1024
    smem_capacity_kib = tpu_info.smem_capacity_bytes // 1024

    if _VMEM_DMA_SIZE_KIB.value is None:
      self.vmem_dma_size_kib = vmem_capacity_kib - 1024  # capacity - 1 MiB
    else:
      self.vmem_dma_size_kib = _VMEM_DMA_SIZE_KIB.value
      if self.vmem_dma_size_kib > vmem_capacity_kib - 1024:
        raise ValueError(
            "VMEM DMA size must be less than or equal to VMEM capacity - 1 MiB."
            " Received: %d KiB"
            % self.vmem_dma_size_kib
        )

    if _SMEM_DMA_SIZE_KIB.value is None:
      self.smem_dma_size_kib = smem_capacity_kib - 16  # capacity - 16 KiB
    else:
      self.smem_dma_size_kib = _SMEM_DMA_SIZE_KIB.value
      if self.smem_dma_size_kib > smem_capacity_kib - 16:
        raise ValueError(
            "SMEM DMA size must be less than or equal to SMEM capacity - 16"
            " KiB. Received: %d KiB"
            % self.smem_dma_size_kib
        )

    self.num_dmas = _NUMBER_OF_DMAS.value
    self.number_of_measurements = _NUMBER_OF_MEASUREMENTS.value

  def _dma_bandwidth_test(
      self, kernel_fn, dma_size_kib, memory_space, num_dmas=1
  ):
    """Execute a given kernel and measure the bandwidth."""
    invocation = pl.pallas_call(
        kernel_fn,
        out_specs=pl.BlockSpec(
            (dma_size_kib, 1024), lambda *args: (0, 0), memory_space=pltpu.HBM
        ),
        out_shape=jax.ShapeDtypeStruct((dma_size_kib, 1024), jnp.uint8),
        scratch_shapes=([
            memory_space((dma_size_kib, 1024), jnp.uint8),
            *[pltpu.SemaphoreType.DMA] * num_dmas,
        ]),
        name=self._KERNEL_NAME,
        interpret=False,
    )

    execution_times_us = self._measure_execution_times(
        invocation, self._KERNEL_NAME
    )
    self._print_bandwidth_statistics(dma_size_kib, num_dmas, execution_times_us)

  def _dma_latency_test(self, kernel_fn, memory_space):
    """Execute a given kernel and measure its latency."""

    invocation = pl.pallas_call(
        kernel_fn,
        out_specs=pl.BlockSpec(
            (128,), lambda *args: (0,), memory_space=pltpu.HBM
        ),
        out_shape=jax.ShapeDtypeStruct((128,), jnp.uint8),
        scratch_shapes=([
            memory_space((128,), jnp.uint8),
            pltpu.SemaphoreType.DMA,
        ]),
        name=self._KERNEL_NAME,
        interpret=False,
    )

    execution_times_us = self._measure_execution_times(
        invocation, self._KERNEL_NAME
    )
    self._print_latency_statistics(execution_times_us)

  def test_vmem_to_hbm_bandwidth_single_dma(self):
    """Measure VMEM to HBM bandwidth using a single DMA transaction."""

    def copy_vmem_to_hbm(hbm_ref, vmem_ref, sem_ref):
      pltpu.async_copy(vmem_ref, hbm_ref, sem_ref).wait()

    self._dma_bandwidth_test(
        copy_vmem_to_hbm, self.vmem_dma_size_kib, pltpu.VMEM
    )

  def test_hbm_to_vmem_bandwidth_single_dma(self):
    """Measure HBM to VMEM bandwidth using a single DMA transaction."""

    def copy_hbm_to_vmem(hbm_ref, vmem_ref, sem_ref):
      pltpu.async_copy(hbm_ref, vmem_ref, sem_ref).wait()

    self._dma_bandwidth_test(
        copy_hbm_to_vmem, self.vmem_dma_size_kib, pltpu.VMEM
    )

  def test_vmem_to_hbm_dma_unloaded_latency(self):
    """Measure the DMA unloaded latency from VMEM to HBM.

    Unloaded latency (or idle latency) represents the minimum theoretical time
    required to complete a DMA transaction under ideal, contention-free
    conditions.
    """

    def copy_vmem_to_hbm(hbm_ref, vmem_ref, sem_ref):
      pltpu.async_copy(vmem_ref, hbm_ref, sem_ref).wait()

    self._dma_latency_test(copy_vmem_to_hbm, pltpu.VMEM)

  def test_hbm_to_vmem_dma_unloaded_latency(self):
    """Measure the DMA unloaded latency from HBM to VMEM.

    Unloaded latency (or idle latency) represents the minimum theoretical time
    required to complete a DMA transaction under ideal, contention-free
    conditions.
    """

    def copy_hbm_to_vmem(hbm_ref, vmem_ref, sem_ref):
      pltpu.async_copy(hbm_ref, vmem_ref, sem_ref).wait()

    self._dma_latency_test(copy_hbm_to_vmem, pltpu.VMEM)

  def test_vmem_to_hbm_bandwidth_multiple_dmas(self):
    """Measure VMEM to HBM bandwidth using multiple DMA transactions."""

    def copy_vmem_to_hbm(hbm_ref, vmem_ref, *sem_ref):
      jobs = [
          pltpu.async_copy(vmem_ref, hbm_ref, sem_ref[i])
          for i in range(self.num_dmas)
      ]
      for job in jobs:
        job.wait()

    self._dma_bandwidth_test(
        copy_vmem_to_hbm, self.vmem_dma_size_kib, pltpu.VMEM, self.num_dmas
    )

  def test_hbm_to_vmem_bandwidth_multiple_dmas(self):
    """Measure HBM to VMEM bandwidth using multiple DMA transactions."""

    def copy_hbm_to_vmem(hbm_ref, vmem_ref, *sem_ref):
      jobs = [
          pltpu.async_copy(hbm_ref, vmem_ref, sem_ref[i])
          for i in range(self.num_dmas)
      ]
      for job in jobs:
        job.wait()

    self._dma_bandwidth_test(
        copy_hbm_to_vmem, self.vmem_dma_size_kib, pltpu.VMEM, self.num_dmas
    )

  def test_smem_to_hbm_bandwidth_single_dma(self):
    """Measure SMEM to HBM bandwidth using a single DMA transaction."""

    def copy_smem_to_hbm(hbm_ref, smem_ref, sem_ref):
      pltpu.async_copy(smem_ref, hbm_ref, sem_ref).wait()

    self._dma_bandwidth_test(
        copy_smem_to_hbm, self.smem_dma_size_kib, pltpu.SMEM
    )

  def test_hbm_to_smem_bandwidth_single_dma(self):
    """Measure HBM to SMEM bandwidth using a single DMA transaction."""

    def copy_hbm_to_smem(hbm_ref, smem_ref, sem_ref):
      pltpu.async_copy(hbm_ref, smem_ref, sem_ref).wait()

    self._dma_bandwidth_test(
        copy_hbm_to_smem, self.smem_dma_size_kib, pltpu.SMEM
    )

  def test_smem_to_hbm_dma_unloaded_latency(self):
    """Measure the DMA unloaded latency from SMEM to HBM.

    Unloaded latency (or idle latency) represents the minimum theoretical time
    required to complete a DMA transaction under ideal, contention-free
    conditions.
    """

    def copy_smem_to_hbm(hbm_ref, smem_ref, sem_ref):
      pltpu.async_copy(smem_ref, hbm_ref, sem_ref).wait()

    self._dma_latency_test(copy_smem_to_hbm, pltpu.SMEM)

  def test_hbm_to_smem_dma_unloaded_latency(self):
    """Measure the DMA unloaded latency from HBM to SMEM.

    Unloaded latency (or idle latency) represents the minimum theoretical time
    required to complete a DMA transaction under ideal, contention-free
    conditions.
    """

    def copy_hbm_to_smem(hbm_ref, smem_ref, sem_ref):
      pltpu.async_copy(hbm_ref, smem_ref, sem_ref).wait()

    self._dma_latency_test(copy_hbm_to_smem, pltpu.SMEM)

  def test_smem_to_hbm_bandwidth_multiple_dmas(self):
    """Measure SMEM to HBM bandwidth using multiple DMA transactions."""

    def copy_smem_to_hbm(hbm_ref, smem_ref, *sem_ref):
      jobs = [
          pltpu.async_copy(smem_ref, hbm_ref, sem_ref[i])
          for i in range(self.num_dmas)
      ]
      for job in jobs:
        job.wait()

    self._dma_bandwidth_test(
        copy_smem_to_hbm, self.smem_dma_size_kib, pltpu.SMEM, self.num_dmas
    )

  def test_hbm_to_smem_bandwidth_multiple_dmas(self):
    """Measure HBM to SMEM bandwidth using multiple DMA transactions."""

    def copy_hbm_to_smem(hbm_ref, smem_ref, *sem_ref):
      jobs = [
          pltpu.async_copy(hbm_ref, smem_ref, sem_ref[i])
          for i in range(self.num_dmas)
      ]
      for job in jobs:
        job.wait()

    self._dma_bandwidth_test(
        copy_hbm_to_smem, self.smem_dma_size_kib, pltpu.SMEM, self.num_dmas
    )


if __name__ == "__main__":
  jax.config.config_with_absl()
  absltest.main()

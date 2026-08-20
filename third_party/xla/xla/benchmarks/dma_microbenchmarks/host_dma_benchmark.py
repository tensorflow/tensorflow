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

"""JAX microbenchmarks for measuring DMA bandwidth between host and TPU."""

from absl import flags
from absl.testing import absltest
import jax
import jax.numpy as jnp
from xla.benchmarks.dma_microbenchmarks import memory_base  # pylint: disable=g-direct-tensorflow-import


_NUMBER_OF_MEASUREMENTS = flags.DEFINE_integer(
    "number_of_measurements",
    default=5,
    help="Number of measurements to take. Default: 5",
)


class HostDmaBenchmarks(memory_base.MemoryBenchmarks):
  """Test suite measuring DMA bandwidth between host memory and TPU memory."""

  def setUp(self):
    super().setUp()
    self.number_of_measurements = _NUMBER_OF_MEASUREMENTS.value

  def test_device_to_host_bandwidth_single_dma(self):
    """Measure device to host bandwidth using a single DMA transaction."""
    tpu = jax.devices()[0]
    device_memory = jax.sharding.SingleDeviceSharding(tpu, memory_kind="device")
    pinned_host_memory = jax.sharding.SingleDeviceSharding(
        tpu, memory_kind="pinned_host"
    )

    dma_size_kib = 1024 * 1024
    arr = jnp.zeros((dma_size_kib, 1024), dtype=jnp.uint8)  # 1GB
    array_on_device = jax.device_put(arr, device_memory)

    @jax.jit
    def copy_device_to_host(x):
      return jax.device_put(x, pinned_host_memory)

    execution_times_us = self._measure_execution_times(
        copy_device_to_host, "jit_copy_device_to_host", array_on_device
    )
    self._print_bandwidth_statistics(dma_size_kib, 1, execution_times_us)

  def test_host_to_device_bandwidth_single_dma(self):
    """Measure host to device bandwidth using a single DMA transaction."""
    tpu = jax.devices()[0]
    device_memory = jax.sharding.SingleDeviceSharding(tpu, memory_kind="device")
    pinned_host_memory = jax.sharding.SingleDeviceSharding(
        tpu, memory_kind="pinned_host"
    )

    dma_size_kib = 1024 * 1024
    arr = jnp.zeros((dma_size_kib, 1024), dtype=jnp.uint8)
    array_on_host = jax.device_put(arr, pinned_host_memory)

    @jax.jit
    def copy_host_to_device(x):
      return jax.device_put(x, device_memory)

    execution_times_us = self._measure_execution_times(
        copy_host_to_device, "jit_copy_host_to_device", array_on_host
    )
    self._print_bandwidth_statistics(dma_size_kib, 1, execution_times_us)


if __name__ == "__main__":
  jax.config.config_with_absl()
  absltest.main()

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

"""JAX microbenchmarks for measuring ICI bandwidth across TPU chips."""

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


class ChipToChipBenchmarks(memory_base.MemoryBenchmarks):
  """Test suite measuring inter-chip interconnect (ICI) bandwidth."""

  def setUp(self):
    super().setUp()
    self.number_of_measurements = _NUMBER_OF_MEASUREMENTS.value

  def get_devices(self):
    """Get a pair of TPU devices on different physical chips."""
    devices = [d for d in jax.devices() if d.core_on_chip == 0]
    if len(devices) < 2:
      self.skipTest("Need at least 2 devices.")
    return devices[:2]

  def test_chip_to_chip_bandwidth(self):
    """Measure inter-chip bandwidth using collective permute."""
    # Create a mesh with 2 devices, each on a different chip.
    devices = self.get_devices()
    mesh = jax.make_mesh((2,), ("chips",), devices=devices)
    partition = jax.sharding.PartitionSpec("chips", None)
    sharding = jax.sharding.NamedSharding(mesh, partition)

    dma_size_kib = 1024 * 1024
    arr = jax.random.randint(
        jax.random.key(0),
        (2, dma_size_kib, 1024),
        minval=0,
        maxval=255,
        dtype=jnp.uint8,
    )
    array_on_devices = jax.device_put(arr, sharding)

    @jax.jit
    def copy_between_chips(x):
      return jax.shard_map(
          lambda x: jax.lax.ppermute(x, "chips", [(0, 1), (1, 0)]),
          mesh=mesh,
          in_specs=partition,
          out_specs=partition
      )(x)

    execution_times_us = self._measure_execution_times(
        copy_between_chips,
        ["collective-permute-start", "collective-permute-done"],
        array_on_devices,
    )
    self._print_bandwidth_statistics(dma_size_kib, 1, execution_times_us)


if __name__ == "__main__":
  jax.config.config_with_absl()
  absltest.main()

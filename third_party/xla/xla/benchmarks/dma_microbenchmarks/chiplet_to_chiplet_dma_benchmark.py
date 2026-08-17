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

"""JAX microbenchmarks for measuring chiplet-to-chiplet bandwidth."""

from absl import flags
from absl.testing import absltest
import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from xla.benchmarks.dma_microbenchmarks import memory_base  # pylint: disable=g-direct-tensorflow-import


_NUMBER_OF_MEASUREMENTS = flags.DEFINE_integer(
    "number_of_measurements",
    default=5,
    help="Number of measurements to take. Default: 5",
)


class ChipletToChipletBenchmarks(memory_base.MemoryBenchmarks):
  """Test suite measuring chiplet-to-chiplet bandwidth."""

  def setUp(self):
    super().setUp()
    self.number_of_measurements = _NUMBER_OF_MEASUREMENTS.value

  def get_devices(self):
    if jax.local_device_count() < 2:
      self.skipTest("Need at least 2 devices.")
    if jax.devices()[0].coords != jax.devices()[1].coords:
      self.skipTest("TPU does not have multiple chiplets.")
    if jax.devices()[0].core_on_chip != 0:
      self.skipTest("Device 0 is not on core 0.")
    if jax.devices()[1].core_on_chip != 1:
      self.skipTest("Device 1 is not on core 1.")
    return jax.devices()[:2]

  def test_chiplet_to_chiplet_bandwidth_with_collective_permute(self):
    """Measure chiplet-to-chiplet bandwidth using collective permute."""

    devices = self.get_devices()
    partition = jax.sharding.PartitionSpec("chiplets", None)
    mesh = jax.make_mesh((2,), ("chiplets",), devices=devices)
    sharding = jax.sharding.NamedSharding(mesh, partition)

    dma_size_kib = 1024 * 1024  # one direction
    arr = jax.random.randint(
        jax.random.key(0), (2, dma_size_kib, 1024), 0, 256, jnp.uint8
    )
    array_on_devices = jax.device_put(arr, sharding)

    @jax.jit
    def copy_between_chiplets(x):
      return jax.shard_map(
          lambda x: jax.lax.ppermute(x, "chiplets", [(0, 1), (1, 0)]),
          mesh=mesh,
          in_specs=partition,
          out_specs=partition,
      )(x)

    execution_times_us = self._measure_execution_times(
        copy_between_chiplets,
        ["collective-permute-start", "collective-permute-done"],
        array_on_devices,
    )
    self._print_bandwidth_statistics(dma_size_kib, 1, execution_times_us)

  def test_chiplet_to_chiplet_bandwidth_with_async_remote_copy(self):
    """Measure chiplet-to-chiplet bandwidth using asynchronous remote copy."""
    devices = self.get_devices()
    partition = jax.sharding.PartitionSpec("chiplets", None)
    mesh = jax.make_mesh((2,), ("chiplets",), devices=devices)
    sharding = jax.sharding.NamedSharding(mesh, partition)

    def copy_from_chiplet_to_chiplet(input_ref, output_ref, send_sem, recv_sem):
      chiplet_id = lax.axis_index("chiplets")

      remote_copy = pltpu.make_async_remote_copy(
          src_ref=input_ref,
          dst_ref=output_ref,
          send_sem=send_sem,
          recv_sem=recv_sem,
          device_id=(1,),
      )

      @pl.when(chiplet_id == 0)
      def send():
        remote_copy.start()
        remote_copy.wait_send()

      @pl.when(chiplet_id == 1)
      def receive():
        remote_copy.wait_recv()

    dma_size_kib = 1024 * 1024
    input_arr = jnp.zeros((2 * dma_size_kib, 1024), jnp.uint8)
    input_arr = jax.device_put(input_arr, sharding)

    kernel = pl.pallas_call(
        copy_from_chiplet_to_chiplet,
        out_shape=jax.ShapeDtypeStruct((dma_size_kib, 1024), jnp.uint8),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=([pltpu.SemaphoreType.DMA] * 2),
        ),
        name=self._KERNEL_NAME,
    )

    jit_kernel = jax.jit(
        jax.shard_map(
            kernel,
            mesh=mesh,
            in_specs=partition,
            out_specs=partition,
            check_vma=False,
        )
    )

    execution_times_us = self._measure_execution_times(
        jit_kernel,
        self._KERNEL_NAME,
        input_arr,
    )
    self._print_bandwidth_statistics(dma_size_kib, 1, execution_times_us)


if __name__ == "__main__":
  jax.config.config_with_absl()
  absltest.main()

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
"""JAX microbenchmark for TPU all-gather collective bandwidth and latency."""

from absl.testing import absltest
import jax
from jax import numpy as jnp
from jax.experimental import shard_map
import numpy as np

from xla.benchmarks.collectives import collectives_benchmark


def zero_crop(x):
  """Prevents x from being a live-out buffer by passing through a ZeroCrop FFI call."""
  if jax.default_backend() == "cpu":
    return x
  return jax.ffi.ffi_call(
      "ZeroCrop",
      result_shape_dtypes=jax.ShapeDtypeStruct(x.shape, x.dtype),
      has_side_effect=True,
  )(x)


class AllGatherBenchmarks(collectives_benchmark.CollectivesBenchmarks):
  """JAX benchmarks for measuring single-tray all-gather bandwidth and latency."""

  def _setup_mesh_and_axis(self):
    """Creates the single-tray device mesh and returns (mesh, ag_axis, num_devices)."""
    devices = self.get_single_tray_devices()
    num_devices = len(devices)

    num_chips = len({tuple(d.coords) for d in devices})
    cores_per_chip = num_devices // num_chips

    if cores_per_chip > 1:
      mesh = jax.sharding.Mesh(
          np.array(devices).reshape((num_chips, cores_per_chip)),
          ("chips", "cores"),
      )
      ag_axis = ("chips", "cores")
    else:
      mesh = jax.sharding.Mesh(np.array(devices), ("chips",))
      ag_axis = "chips"

    return mesh, ag_axis, num_devices

  def _run_all_gather_benchmark(
      self,
      global_shape,
      in_partition,
      gather_axis,
      op_name,
  ):
    """Runs a single-HLO all-gather benchmark and logs bandwidth/latency stats."""
    mesh, ag_axis, num_devices = self._setup_mesh_and_axis()
    out_partition = jax.sharding.PartitionSpec(*([None] * len(global_shape)))

    sharding = jax.sharding.NamedSharding(mesh, in_partition)
    arr = jax.random.uniform(
        jax.random.key(0),
        global_shape,
        dtype=jnp.bfloat16,
    )
    array_on_devices = jax.device_put(arr, sharding)
    per_device_bytes = arr.nbytes // num_devices

    @jax.jit
    def all_gather_fn(x):
      return shard_map.shard_map(
          lambda x_shard: zero_crop(
              jax.lax.all_gather(x_shard, ag_axis, axis=gather_axis)
          ),
          mesh=mesh,
          in_specs=in_partition,
          out_specs=out_partition,
          check_rep=False,
      )(x)

    # Warmup run to compile and initialize buffers before tracing.
    warmup_res = all_gather_fn(array_on_devices)
    jax.block_until_ready(warmup_res)

    execution_times_us = self._measure_execution_times(
        all_gather_fn,
        ["all-gather"],
        array_on_devices,
    )
    self._print_all_gather_bandwidth_statistics(
        per_device_size_bytes=per_device_bytes,
        num_devices=num_devices,
        latencies_us=execution_times_us,
        op_name=op_name,
    )

  def _print_all_gather_bandwidth_statistics(
      self,
      per_device_size_bytes,
      num_devices,
      latencies_us,
      op_name="all-gather",
  ):
    """Computes all-gather data sizes and prints bandwidth/latency statistics."""
    total_gathered_bytes = per_device_size_bytes * num_devices
    # In ring-based all-gather, each device transmits (N-1)/N of the
    # gathered data.
    bus_data_bytes = per_device_size_bytes * (num_devices - 1)
    self._print_collective_bandwidth_statistics(
        op_name=op_name,
        per_device_size_bytes=per_device_size_bytes,
        total_size_bytes=total_gathered_bytes,
        bus_data_bytes=bus_data_bytes,
        num_devices=num_devices,
        latencies_us=latencies_us,
    )

  def _get_per_device_chunk_mib(self, num_devices):
    """Returns per-device chunk size in MiB from collective_size_mib and num_devices."""
    if self.collective_size_mib % num_devices != 0:
      raise ValueError(
          f"collective_size_mib ({self.collective_size_mib}) must be divisible"
          f" by num_devices ({num_devices})."
      )
    return self.collective_size_mib // num_devices

  def test_all_gather_major_dim_bandwidth(self):
    _, ag_axis, num_devices = self._setup_mesh_and_axis()
    chunk_size_mib = self._get_per_device_chunk_mib(num_devices)
    self._run_all_gather_benchmark(
        global_shape=(num_devices, chunk_size_mib * 512, 1024),
        in_partition=jax.sharding.PartitionSpec(ag_axis, None, None),
        gather_axis=0,
        op_name="all-gather-major-dim",
    )

  def test_all_gather_minor_dim_bandwidth(self):
    _, ag_axis, num_devices = self._setup_mesh_and_axis()
    chunk_size_mib = self._get_per_device_chunk_mib(num_devices)
    self._run_all_gather_benchmark(
        global_shape=(chunk_size_mib, num_devices * 512, 1024),
        in_partition=jax.sharding.PartitionSpec(None, ag_axis, None),
        gather_axis=1,
        op_name="all-gather-minor-dim",
    )


if __name__ == "__main__":
  jax.config.config_with_absl()
  absltest.main()

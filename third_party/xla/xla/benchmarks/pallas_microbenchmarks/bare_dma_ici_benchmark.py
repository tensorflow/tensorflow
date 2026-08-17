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

"""Bare inter-chip DMA ICI bandwidth microbenchmark without HLO collectives."""

from absl import app
from absl import flags
from absl import logging
import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import pandas as pd

from xla.benchmarks.jax_microbenchmarks import jax_profiler_utils  # pylint: disable=g-direct-tensorflow-import
from xla.benchmarks.pallas_microbenchmarks import memory_utils

_SIZE_MIB = flags.DEFINE_integer(
    "size_mib",
    128,
    "Size of the bare DMA payload in MiB.",
)
_REPEAT = flags.DEFINE_integer(
    "repeat",
    2,
    "Number of times the bare DMA kernel is repeated per run.",
)
_RUNS = flags.DEFINE_integer(
    "runs",
    1,
    "Number of end-to-end bare DMA runs.",
)


def run_bare_dma_ici_benchmark(
    size_mib: int,
    repeat: int,
    runs: int,
) -> pd.DataFrame | None:
  """Runs a bare inter-chip DMA ICI bandwidth test using Pallas make_async_remote_copy."""
  devices = jax.devices()
  logging.info("Found %d JAX device(s): %s", len(devices), devices)

  # Determine how many cores are inside a single chip by looking at coordinates.
  cores_per_chip = len([d for d in devices if d.coords == devices[0].coords])

  if len(devices) < cores_per_chip + 1:
    raise ValueError(
        "Need at least 2 distinct chips (over multiple cores) for ICI."
    )

  # Extract the first device of chip 1 and the first device of chip 2.
  devices_pair = [devices[0], devices[cores_per_chip]]

  size_bytes = size_mib * 1024 * 1024
  num_elements = size_bytes // 4
  shape = (num_elements,)

  logging.info(
      "Running bare inter-chip DMA between %s <-> %s",
      devices_pair[0],
      devices_pair[1],
  )

  mesh = jax.make_mesh((2,), ("chips",), devices=devices_pair)
  partition = jax.sharding.PartitionSpec("chips")
  sharding = jax.sharding.NamedSharding(mesh, partition)

  global_shape = (2 * num_elements,)
  arr_global = jnp.ones(global_shape, dtype=jnp.uint32)
  input_arr = jax.device_put(arr_global, sharding)

  kernel_name = "bare_ici_dma_kernel"
  pallas_kernel = pl.pallas_call(
      memory_utils.remote_dma_kernel,
      out_shape=jax.ShapeDtypeStruct(shape, jnp.uint32),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          in_specs=[
              pl.BlockSpec(memory_space=pltpu.HBM),
          ],
          out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
          scratch_shapes=[
              pltpu.SemaphoreType.DMA(()),
              pltpu.SemaphoreType.DMA(()),
          ],
      ),
      name=kernel_name,
  )

  @jax.jit
  def run_bare_dma(arr):
    return jax.shard_map(
        pallas_kernel,
        mesh=mesh,
        in_specs=partition,
        out_specs=partition,
        check_vma=False,
    )(arr)

  f_compiled = run_bare_dma.lower(input_arr).compile()

  all_results = []
  for run_idx in range(runs):
    with jax_profiler_utils.JaxProfiler(kernel_name) as profiler:
      res = f_compiled(input_arr)
      res.block_until_ready()
      for _ in range(repeat - 1):
        res = f_compiled(input_arr)
        res.block_until_ready()

    if profiler.result is not None and profiler.result.runtimes_us:
      df = profiler.result.as_dataframe()
      df["bandwidth_gbps"] = size_bytes / (df["runtime_us"].astype(float) * 1e3)
      df["run_idx"] = run_idx
      df["size_mib"] = size_mib
      all_results.append(df)
      summary_str = df[["runtime_us", "bandwidth_gbps"]].to_string()
      logging.info(
          "Run %d Profiler results (%s):\n%s", run_idx, kernel_name, summary_str
      )
    else:
      logging.warning(
          "Run %d: No profiler duration captured for kernel '%s'",
          run_idx,
          kernel_name,
      )

  if all_results:
    combined_df = pd.DataFrame(pd.concat(all_results, ignore_index=True))
    avg_bw = combined_df["bandwidth_gbps"].mean()
    std_bw = combined_df["bandwidth_gbps"].std()
    summary_box = (
        "\n================ BARE DMA ICI BANDWIDTH SUMMARY ================\n"
        f"Kernel     : {kernel_name}\n"
        f"Payload    : {size_mib} MiB ({size_bytes} bytes)\n"
        f"Devices    : {len(devices)} device(s)\n"
        f"Bandwidth  : {avg_bw:.2f} +/- {std_bw:.2f} GB/s\n"
        "================================================================\n"
    )
    logging.info("%s", summary_box)
    return combined_df
  return None


def main(_):
  logging.set_verbosity(logging.INFO)
  run_bare_dma_ici_benchmark(
      size_mib=_SIZE_MIB.value,
      repeat=_REPEAT.value,
      runs=_RUNS.value,
  )


if __name__ == "__main__":
  app.run(main)

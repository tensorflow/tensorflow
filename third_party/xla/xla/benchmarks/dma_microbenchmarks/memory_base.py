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

"""Base classes and utilities for JAX memory microbenchmarks on TPUs."""

import gzip
import json
import logging
import pathlib
import tempfile

from absl import flags
from absl.testing import absltest
import jax
import jax.experimental.pallas.tpu as pltpu
import numpy as np

from xla.benchmarks.core import benchmark
from xla.benchmarks.jax_microbenchmarks import jax_profiler_utils


_NUMBER_OF_MEASUREMENTS = flags.DEFINE_integer(
    "number_of_measurements",
    default=5,
    help="Number of measurements to take. Default: 5",
)


class MemoryBenchmarks(absltest.TestCase):
  """Base test case providing profiling and bandwidth metric helpers."""

  _KERNEL_NAME = "memory_benchmark_kernel"

  def setUp(self):
    super().setUp()
    if not any(device.platform == "tpu" for device in jax.devices()):
      self.skipTest("This test requires TPU hardware.")
    self.number_of_measurements = _NUMBER_OF_MEASUREMENTS.value
    self.latencies_us = []
    self.metrics = {}

  def _normalize_kernel_names(self, kernel_names):
    """Normalize kernel names into a list of strings."""
    if isinstance(kernel_names, str):
      return [kernel_names]
    elif isinstance(kernel_names, tuple):
      return list(kernel_names)
    elif isinstance(kernel_names, list):
      return kernel_names
    raise TypeError(
        "`kernel_names` must be a string, tuple, or list of strings. "
        f"Got: {type(kernel_names)}"
    )

  def _extract_execution_times(self, profiler_dir, kernel_names):
    """Extract the execution times from the profiler data."""
    trace_files = list(pathlib.Path(profiler_dir).glob("**/*.trace.json.gz"))
    if not trace_files:
      raise FileNotFoundError("Could not find trace.json.gz")

    kernel_dur_per_core = {}
    for trace_file in trace_files:
      with gzip.open(trace_file, "rt") as f:
        trace_events = json.load(f).get("traceEvents", [])

      for event in trace_events:
        for kernel_name in kernel_names:
          if event.get("name", "").startswith(kernel_name) and "dur" in event:
            pid = event.get("pid")
            if pid not in kernel_dur_per_core:
              kernel_dur_per_core[pid] = {name: [] for name in kernel_names}
            kernel_dur_per_core[pid][kernel_name].append(event["dur"])

    execution_times_us = []
    for _, pid_durs in kernel_dur_per_core.items():
      # Group the durations of the kernels by pid (i.e., per core)
      zipped = zip(*[pid_durs[name] for name in kernel_names], strict=True)
      # Append the durations of all cores to the list
      execution_times_us.extend([sum(durations) for durations in zipped])

    if not execution_times_us:
      raise ValueError("Could not find execution times.")
    return execution_times_us

  def _measure_execution_times(
      self,
      kernel,
      kernel_names,
      *kernel_args,
      **kernel_kwargs,
  ):
    """Collect XProf measurements using the JAX Profiler API."""
    kernel_names = self._normalize_kernel_names(kernel_names)

    with tempfile.TemporaryDirectory() as tmpdir:
      with jax.profiler.trace(tmpdir):
        for _ in range(self.number_of_measurements):
          result = kernel(*kernel_args, **kernel_kwargs)
          jax.block_until_ready(result)
      return self._extract_execution_times(tmpdir, kernel_names)

  def _print_bandwidth_statistics(self, dma_size_kib, num_dmas, latencies_us):
    """Print the bandwidth statistics for a given test."""
    latencies_ns = np.array(latencies_us) * 1e3
    total_size_bytes = dma_size_kib * num_dmas * 1024
    # Bandwidth unit is GB/s because the unit of latencies is nanoseconds.
    bandwidth_gbps = total_size_bytes / latencies_ns

    avg_bandwidth_gbps = np.mean(bandwidth_gbps)
    std_bandwidth_gbps = np.std(bandwidth_gbps)
    tpu_info = pltpu.get_tpu_info()
    vmem_capacity_mib = tpu_info.vmem_capacity_bytes / (2**20)

    logging.info("Test: %s", self._testMethodName)
    logging.info("\tTPU generation: %s", tpu_info.chip_version)
    logging.info("\tVMEM capacity: %.1f MiB", vmem_capacity_mib)
    if dma_size_kib > 1024:
      logging.info("\tDMA size: %.1f MiB", (dma_size_kib / 1024))
    else:
      logging.info("\tDMA size: %.1f KiB", dma_size_kib)
    logging.info("\tNumber of DMAs: %d", num_dmas)
    if total_size_bytes > 1024**2:
      logging.info("\tCopied: %.1f MiB", (total_size_bytes / (1024**2)))
    else:
      logging.info("\tCopied: %.1f KiB", (total_size_bytes / 1024))
    logging.info(
        "\tBandwidth: %.2f +/- %.2f GB/s",
        avg_bandwidth_gbps,
        std_bandwidth_gbps,
    )

    self.latencies_us = list(latencies_us)
    self.metrics = {
        "dma_size_kib": dma_size_kib,
        "num_dmas": num_dmas,
        "bandwidth_gbps": float(avg_bandwidth_gbps),
    }

  def _print_latency_statistics(self, latencies_us):
    """Print the latency statistics for a given test."""
    latencies_ns = np.array(latencies_us) * 1e3
    avg_latency_ns = np.mean(latencies_ns)
    std_latency_ns = np.std(latencies_ns)
    tpu_info = pltpu.get_tpu_info()
    logging.info("Test: %s", self._testMethodName)
    logging.info("\tTPU generation: %s", tpu_info.chip_version)
    logging.info("\tLatency: %.2f +/- %.2f ns", avg_latency_ns, std_latency_ns)

    self.latencies_us = list(latencies_us)


class DmaBenchmarkConfig(benchmark.BenchmarkConfig):
  """Config identifying a single DMA test method for the benchmark driver."""

  def __init__(self, suite, test_class, test_name):
    self.suite = suite
    self.test_class = test_class
    self.test_name = test_name
    self.metrics = {}

  def as_dict(self):
    return {"suite": self.suite, "test": self.test_name, **self.metrics}

  def get_benchmark(self):
    return DmaBenchmark(self)


class DmaBenchmark(benchmark.Benchmark):
  """Runs a DMA test method and reports its latencies to the driver."""

  def __init__(self, config):
    self._config = config

  def get_input_shapes_and_dtypes(self):
    return []

  def target_fn(self):
    return lambda: None

  def kernel_name(self):
    return self._config.test_name

  def run(self, **kwargs):
    del kwargs
    test = self._config.test_class(self._config.test_name)
    try:
      test.setUp()
      getattr(test, self._config.test_name)()
    except absltest.SkipTest as e:
      logging.warning("Skipping %s: %s", self._config.test_name, e)
      return [None]
    self._config.metrics.update(test.metrics)
    return [jax_profiler_utils.JaxProfilerResult(runtimes_us=test.latencies_us)]

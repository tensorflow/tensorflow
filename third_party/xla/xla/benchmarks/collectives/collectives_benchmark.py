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
"""Base class and utilities for JAX collective microbenchmarks."""

import gzip
import json
import os
import pathlib
import shutil
import sys
import tempfile

from absl import flags
from absl import logging
from absl.testing import absltest
import immutabledict
import jax
import jax.experimental.pallas.tpu as pltpu
import numpy as np

from xla.benchmarks.core import benchmark
from xla.benchmarks.jax_microbenchmarks import jax_profiler_utils

_NUMBER_OF_MEASUREMENTS = flags.DEFINE_integer(
    "number_of_measurements",
    default=5,
    help="Number of measurements to take. Default: 5",
    allow_override=True,
)

_COLLECTIVE_SIZE_MIB = flags.DEFINE_integer(
    "collective_size_mib",
    default=1024,
    help="Total collective size in MiB across all devices. Default: 1024",
)

EXPECTED_CHIPS_PER_HOST = immutabledict.immutabledict({
    pltpu.ChipVersion.TPU_V4: 4,
    pltpu.ChipVersion.TPU_V4I: 4,
    pltpu.ChipVersion.TPU_V5E: 4,
    pltpu.ChipVersion.TPU_V5P: 4,
    pltpu.ChipVersion.TPU_V6E: 4,
    pltpu.ChipVersion.TPU_7: 4,
    pltpu.ChipVersion.TPU_7X: 4,
    pltpu.ChipVersion.TPU_8I: 2,
    pltpu.ChipVersion.TPU_8T: 2,
})


class CollectivesBenchmarks(absltest.TestCase):
  """Base benchmark class for measuring JAX collective operations."""

  def setUp(self):
    super().setUp()
    if not any(device.platform == "tpu" for device in jax.devices()):
      self.skipTest("This test requires TPU hardware.")
    self.number_of_measurements = _NUMBER_OF_MEASUREMENTS.value
    self.collective_size_mib = _COLLECTIVE_SIZE_MIB.value
    self.latencies_us = []
    self.metrics = {}

  def verify_single_tray_setup(self):
    """Verifies that the test is running on exactly one full TPU tray."""
    if not any(device.platform == "tpu" for device in jax.devices()):
      self.skipTest("This test requires TPU hardware.")

    if jax.process_count() > 1:
      self.skipTest(
          "Single-host test expected process_count=1, got"
          f" {jax.process_count()}."
      )

    local_devices = jax.local_devices()
    all_devices = jax.devices()
    if len(local_devices) != len(all_devices):
      self.skipTest(
          f"Not all devices are local to this host: local={len(local_devices)},"
          f" all={len(all_devices)}."
      )

    tpu_info = pltpu.get_tpu_info()
    chip_version = tpu_info.chip_version
    if not isinstance(chip_version, pltpu.ChipVersion):
      logging.warning(
          "Unknown TPU generation %s for chip count verification.", chip_version
      )
      return
    expected_chips = EXPECTED_CHIPS_PER_HOST.get(chip_version)
    if expected_chips is None:
      logging.warning(
          "Unknown TPU generation %s for chip count verification.", chip_version
      )
      return

    # Check unique chip coordinates across local TPU devices.
    coords = [tuple(d.coords) for d in local_devices]
    num_unique_chips = len(set(coords))
    if num_unique_chips != expected_chips:
      self.skipTest(
          f"Expected {expected_chips} TPU chips for a single {chip_version}"
          f" tray, but found {num_unique_chips}."
      )

    # Check device count based on cores per chip.
    num_cores_per_chip = chip_version.num_physical_tensor_cores_per_chip
    expected_devices_split = expected_chips * num_cores_per_chip
    expected_devices_megacore = expected_chips

    if len(local_devices) not in (
        expected_devices_split,
        expected_devices_megacore,
    ):
      self.skipTest(
          f"Expected {expected_devices_split} (or {expected_devices_megacore}) "
          f"devices for a single {chip_version} tray, got "
          f"{len(local_devices)}."
      )

  def get_single_tray_devices(self):
    """Verifies the single-tray setup and returns the list of local devices."""
    self.verify_single_tray_setup()
    return jax.local_devices()

  def _normalize_kernel_names(self, kernel_names):
    """Normalizes kernel names into a list of strings."""
    if isinstance(kernel_names, str):
      return [kernel_names]
    elif isinstance(kernel_names, (tuple, list)):
      return list(kernel_names)
    raise TypeError(
        "`kernel_names` must be a string, tuple, or list of strings. "
        f"Got: {type(kernel_names)}"
    )

  def _extract_execution_times(self, profiler_dir, kernel_names):
    """Extracts execution times from profiler data for matching kernel names."""
    undeclared_outputs_dir = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
    if undeclared_outputs_dir:
      test_name = self._testMethodName
      for profile_file in pathlib.Path(profiler_dir).rglob("*"):
        if profile_file.is_file() and (
            profile_file.name.endswith(".xplane.pb")
            or profile_file.name.endswith(".trace.json.gz")
        ):
          dest = (
              pathlib.Path(undeclared_outputs_dir)
              / f"{test_name}_{profile_file.name}"
          )
          shutil.copy2(profile_file, dest)
          logging.info(
              "Saved XProf profile artifact to undeclared outputs: %s", dest
          )

    trace_files = list(pathlib.Path(profiler_dir).glob("**/*.trace.json.gz"))
    if not trace_files:
      raise FileNotFoundError(f"Could not find trace.json.gz in {profiler_dir}")

    kernel_durs_per_core = {}
    sc_kernel_durs_per_core = {}
    for trace_file in trace_files:
      with gzip.open(trace_file, "rt") as f:
        trace_events = json.load(f).get("traceEvents", [])

      has_sc_offload = any(
          e.get("args", {}).get("hlo_category") == "async-done"
          and e.get("args", {}).get("offload_type") == "OFFLOAD_COLLECTIVE"
          for e in trace_events
      )
      matched_count = 0
      sample_events = []
      for event in trace_events:
        event_name = event.get("name", "")
        args = event.get("args", {})
        tf_op = args.get("tf_op", "")
        hlo_op = args.get("hlo_op", "")
        if "dur" in event:
          is_host_wrapper = event_name.startswith((
              "PjitFunction",
              "CommonPjRt",
              "PJRT_",
              "PjRt",
              "$",
              "Mutex",
              "PythonRefManager",
              "ParseArguments",
              "Wait for ",
          ))
          if len(sample_events) < 30 and event_name and not is_host_wrapper:
            sample_events.append(
                f"pid={event.get('pid')} name='{event_name}'"
                f" tf_op='{tf_op}' hlo_op='{hlo_op}'"
            )
          if is_host_wrapper:
            continue
          hlo_category = args.get("hlo_category", "")
          if hlo_category in ("async-start", "async-update"):
            continue
          offload_type = args.get("offload_type", "")
          is_sc_offload = (
              hlo_category == "async-done"
              and offload_type == "OFFLOAD_COLLECTIVE"
          )
          for kernel_name in kernel_names:
            k_hyphen = kernel_name.replace("_", "-")
            k_under = kernel_name.replace("-", "_")
            is_name_match = event_name.startswith((
                k_hyphen,
                k_under,
                "ragged-all-to-all",
            ))
            if has_sc_offload:
              if is_sc_offload:
                pid = event.get("pid")
                if pid not in kernel_durs_per_core:
                  kernel_durs_per_core[pid] = []
                kernel_durs_per_core[pid].append(event["dur"])
                matched_count += 1
                logging.info(
                    "[Trace Event] pid=%s, dur=%.2f us, name='%s', tf_op='%s',"
                    " hlo_op='%s', hlo_category='%s', offload_type='%s'",
                    pid,
                    event["dur"],
                    event_name,
                    tf_op,
                    hlo_op,
                    hlo_category,
                    offload_type,
                )
                break
              elif is_name_match and not offload_type:
                pid = event.get("pid")
                if pid not in sc_kernel_durs_per_core:
                  sc_kernel_durs_per_core[pid] = []
                sc_kernel_durs_per_core[pid].append(event["dur"])
                break
            elif is_name_match:
              pid = event.get("pid")
              if pid not in kernel_durs_per_core:
                kernel_durs_per_core[pid] = []
              kernel_durs_per_core[pid].append(event["dur"])
              matched_count += 1
              logging.info(
                  "[Trace Event] pid=%s, dur=%.2f us, name='%s', tf_op='%s',"
                  " hlo_op='%s', hlo_category='%s', offload_type='%s'",
                  pid,
                  event["dur"],
                  event_name,
                  tf_op,
                  hlo_op,
                  hlo_category,
                  offload_type,
              )
              break
      if matched_count == 0:
        logging.warning(
            "No events matched %s in %s. Non-host sample trace events with"
            " dur: %s",
            kernel_names,
            trace_file.name,
            sample_events,
        )
      logging.info(
          "Processed trace file %s: found %d matching events across %d"
          " pids/cores.",
          trace_file.name,
          matched_count,
          len(kernel_durs_per_core),
      )

    if sc_kernel_durs_per_core:
      sc_durs = []
      for pid, durs in sc_kernel_durs_per_core.items():
        sc_durs.extend(durs)
      if sc_durs:
        logging.info(
            "SparseCore pure kernel execution durations across %d PIDs (%d"
            " samples): %.2f +/- %.2f us",
            len(sc_kernel_durs_per_core),
            len(sc_durs),
            np.mean(sc_durs),
            np.std(sc_durs),
        )

    execution_times_us = []
    for pid, durs in kernel_durs_per_core.items():
      if durs:
        logging.info("PID %s event durations: %s", pid, durs)
        num_measurements = self.number_of_measurements
        if len(durs) >= num_measurements and len(durs) % num_measurements == 0:
          k = len(durs) // num_measurements
          for i in range(0, len(durs), k):
            execution_times_us.append(sum(durs[i : i + k]))
        else:
          execution_times_us.extend(durs)

    if not execution_times_us:
      raise ValueError(
          f"Could not find execution times matching {kernel_names}."
      )
    return execution_times_us

  def _measure_execution_times(
      self,
      kernel,
      kernel_names,
      *kernel_args,
      **kernel_kwargs,
  ):
    """Collect XProf measurements using the JAX Profiler API."""
    normalized_names = self._normalize_kernel_names(kernel_names)

    # Warmup run to compile and initialize buffers before tracing.
    warmup_res = kernel(*kernel_args, **kernel_kwargs)
    jax.block_until_ready(warmup_res)

    with tempfile.TemporaryDirectory() as tmpdir:
      with jax.profiler.trace(tmpdir):
        for _ in range(self.number_of_measurements):
          result = kernel(*kernel_args, **kernel_kwargs)
          jax.block_until_ready(result)
      return self._extract_execution_times(tmpdir, normalized_names)

  def _print_collective_bandwidth_statistics(
      self,
      op_name,
      per_device_size_bytes,
      total_size_bytes,
      bus_data_bytes,
      num_devices,
      latencies_us,
  ):
    """Prints bandwidth and latency statistics for a collective operation."""
    latencies_ns = np.array(latencies_us) * 1e3
    bus_bandwidth_gbps = bus_data_bytes / latencies_ns
    effective_bandwidth_gbps = total_size_bytes / latencies_ns

    avg_bus_bw = float(np.mean(bus_bandwidth_gbps))
    std_bus_bw = float(np.std(bus_bandwidth_gbps))
    avg_eff_bw = float(np.mean(effective_bandwidth_gbps))
    std_eff_bw = float(np.std(effective_bandwidth_gbps))
    avg_latency_us = float(np.mean(latencies_us))
    std_latency_us = float(np.std(latencies_us))

    tpu_info = pltpu.get_tpu_info()

    coords = [tuple(getattr(d, "coords", (0,))) for d in jax.local_devices()]
    mesh_dims = "x".join(str(max(dim) + 1) for dim in zip(*coords))
    num_chips = len(set(coords))
    cores_per_chip = num_devices // num_chips if num_chips else 1
    topology_str = (
        f"{mesh_dims} mesh ({num_chips} chips x {cores_per_chip} cores)"
    )

    flags_list = [arg for arg in sys.argv[1:] if arg.startswith("--xla_")]
    flags_str = " ".join(flags_list) if flags_list else "default"

    logging.info("==================================================")
    logging.info("Test: %s", self._testMethodName)
    logging.info("\tCollective Op: %s", op_name)
    logging.info("\tTPU generation: %s", tpu_info.chip_version)
    logging.info("\tNumber of devices: %d", num_devices)
    logging.info("\tTopology: %s", topology_str)
    logging.info("\tFlags: %s", flags_str)
    if per_device_size_bytes >= 1024**2:
      logging.info(
          "\tPer-device chunk size: %.2f MiB",
          per_device_size_bytes / (1024**2),
      )
    else:
      logging.info(
          "\tPer-device chunk size: %.2f KiB", per_device_size_bytes / 1024
      )
    if total_size_bytes >= 1024**2:
      logging.info(
          "\tTotal collective size: %.2f MiB", total_size_bytes / (1024**2)
      )
    else:
      logging.info(
          "\tTotal collective size: %.2f KiB", total_size_bytes / 1024
      )
    logging.info("\tLatency: %.2f +/- %.2f us", avg_latency_us, std_latency_us)
    logging.info("\tBus Bandwidth: %.2f +/- %.2f GB/s", avg_bus_bw, std_bus_bw)
    logging.info(
        "\tEffective Bandwidth: %.2f +/- %.2f GB/s", avg_eff_bw, std_eff_bw
    )
    logging.info("==================================================")

    self.latencies_us = [float(x) for x in latencies_us]
    self.metrics = {
        "per_device_size_mib": float(per_device_size_bytes / (1024**2)),
        "total_size_mib": float(total_size_bytes / (1024**2)),
        "num_devices": num_devices,
        "bus_bandwidth_gbps": avg_bus_bw,
        "effective_bandwidth_gbps": avg_eff_bw,
    }


class CollectiveBenchmarkConfig(benchmark.BenchmarkConfig):
  """Config identifying a single collective test method for the benchmark driver."""

  def __init__(self, suite, test_class, test_name):
    self.suite = suite
    self.test_class = test_class
    self.test_name = test_name
    self.metrics = {}

  def as_dict(self):
    return {"suite": self.suite, "test": self.test_name, **self.metrics}

  def get_benchmark(self):
    return CollectiveBenchmark(self)


class CollectiveBenchmark(benchmark.Benchmark):
  """Runs a collective test method and reports its latencies to the driver."""

  def __init__(self, config):
    self._config = config

  # Required by benchmark.Benchmark ABC; unused since run() delegates to test.
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


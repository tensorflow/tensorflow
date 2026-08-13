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

"""Utilities for JAX Profiler tracing in benchmarks."""

import glob
import gzip
import json
import os
import tempfile
import types
from typing import Sequence
from absl import logging
import jax
import pandas as pd


class JaxProfilerResult:
  """Holds the result of JAX profiling."""

  def __init__(
      self, runtimes_us: Sequence[float] = (), flops: float = 0.0
  ) -> None:
    self.runtimes_us = runtimes_us
    self.flops = flops

  def as_dataframe(self) -> pd.DataFrame:
    return pd.DataFrame({
        "runtime_us": self.runtimes_us,
        "flops": self.flops,
    })


class JaxProfiler:
  """Context manager for JAX Profiler."""

  kernel_name: str
  temp_dir: tempfile.TemporaryDirectory[str] | None
  result: JaxProfilerResult

  def __init__(self, kernel_name: str) -> None:
    self.kernel_name = kernel_name
    self.temp_dir = None
    self.result = JaxProfilerResult()

  def __enter__(self) -> "JaxProfiler":
    self.temp_dir = tempfile.TemporaryDirectory()
    logging.debug("Starting JAX trace to %s", self.temp_dir.name)
    try:
      jax.profiler.start_trace(self.temp_dir.name)
    except Exception as e:  # pylint: disable=broad-except
      logging.warning("Failed to start JAX trace: %s", e)
      self.temp_dir.cleanup()
      self.temp_dir = None
    return self

  def __exit__(
      self,
      exc_type: type[BaseException] | None,
      exc_val: BaseException | None,
      exc_tb: types.TracebackType | None,
  ) -> None:
    if self.temp_dir is not None:
      try:
        jax.profiler.stop_trace()
        trace_files = glob.glob(
            os.path.join(self.temp_dir.name, "**/*.trace.json.gz"),
            recursive=True,
        )
        if trace_files:
          trace_file = trace_files[0]
          with gzip.open(trace_file, "rb") as f:
            trace_data = json.load(f)

          if "traceEvents" in trace_data:
            events = trace_data["traceEvents"]
            matching_events = [
                e
                for e in events
                if (
                    e.get("name", "").startswith(self.kernel_name)
                    or self.kernel_name in e.get("args", {}).get("tf_op", "")
                )
            ]

            times_us = [e["dur"] for e in matching_events if "dur" in e]
            self.result.runtimes_us = times_us
            if times_us:
              flops = 0.0
              for e in matching_events:
                flops_str = e.get("args", {}).get("model_flops")
                if flops_str:
                  try:
                    flops = float(flops_str)
                    break
                  except ValueError:
                    pass
              self.result.flops = flops
            else:
              logging.error(
                  "jax_profiler: No events with duration found matching %s",
                  self.kernel_name,
              )
          else:
            logging.error("No traceEvents found in trace data")
        else:
          logging.error("No trace.json.gz file found")
      except Exception:  # pylint: disable=broad-except
        logging.exception("Failed to parse JAX trace")
        raise
      finally:
        self.temp_dir.cleanup()

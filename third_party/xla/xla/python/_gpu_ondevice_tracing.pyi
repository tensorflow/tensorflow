# Copyright 2025 The OpenXLA Authors.
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
# ==============================================================================
"""Utilities for GPU on-device tracing inject events into Xprof profiler."""


def active_version() -> int:
  """Returns the active version of the GPU on-device tracing.

  0 mean no active tracing.
  """
  ...


def start_injection_instance(version: int) -> int:
  """Starts a new injection instance of the GPU on-device tracing."""
  ...


def inject(
    version: int,
    injection_instance_id: int,
    tag_name: str,
    tag_id: int,
    pid: int,
    tid: int,
    start_time_ns: int,
    duration_ps: int,
) -> None:
  """Injects an event into the Xprof profiler."""
  ...

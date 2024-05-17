# Copyright 2022 The OpenXLA Authors.
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

from typing import Any, List, Optional

class TransferGuardLevel:
  ALLOW: Any
  LOG: Any
  DISALLOW: Any
  LOG_EXPLICIT: Any
  DISALLOW_EXPLICIT: Any

class TransferGuardState:
  host_to_device: Optional[TransferGuardLevel]
  device_to_device: Optional[TransferGuardLevel]
  device_to_host: Optional[TransferGuardLevel]

  explicit_device_put: bool
  explicit_device_get: bool

def global_state() -> TransferGuardState: ...
def thread_local_state() -> TransferGuardState: ...

class _TestingScopedLogSink:
  def __enter__(self) -> _TestingScopedLogSink: ...
  def __exit__(self, *args, **kwargs) -> None: ...
  def logs(self) -> List[str]: ...

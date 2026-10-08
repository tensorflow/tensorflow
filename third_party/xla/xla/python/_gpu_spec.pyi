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

import enum

class GpuTargetConfig:
  @property
  def platform_name(self) -> str: ...
  @property
  def device_description_str(self) -> str: ...
  @property
  def arch_name(self) -> str: ...
  @property
  def compute_capability(self) -> int: ...
  @property
  def core_count(self) -> int: ...
  @property
  def smem_capacity_bytes(self) -> int: ...

class GpuModel(enum.Enum):
  A100_PCIE_80 = 0

  A100_SXM_40 = 1

  A100_SXM_80 = 2

  A6000 = 3

  B200 = 4

  B300 = 6

  BMG_G21 = 7

  H100_PCIE = 8

  H100_SXM = 9

  H200 = 11

  MI200 = 12

  P100 = 14

  PVC = 15

  VR_NVL72 = 16

  V100 = 17

  GB200 = 18

  GB300 = 19

  RTX6000PRO = 20

  GFX1250 = 21

  MI350 = 13

def get_gpu_spec(gpu_model: GpuModel) -> GpuTargetConfig: ...

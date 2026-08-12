# Copyright 2026 The OpenXLA Authors. All Rights Reserved.
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
# =============================================================================

"""Maps GPU target names to their configuration files.

This file contains the `target_configs` dictionary, which provides a mapping
from common GPU model names (e.g., "h100", "a100", "b200") to the
build labels of their corresponding GpuTargetConfigProto textproto files.

This allows build rules like `xla_aot_compile_gpu` to easily reference
target configurations using short, human-readable names.
"""

load("//xla/tsl:package_groups.bzl", "DEFAULT_LOAD_VISIBILITY")

visibility(DEFAULT_LOAD_VISIBILITY)

target_config_map = {
    "a100_pcie_80": "//xla/backends/gpu/target_config:specs/a100_pcie_80.txtpb",
    "a100_sxm_40": "//xla/backends/gpu/target_config:specs/a100_sxm_40.txtpb",
    "a100_sxm_80": "//xla/backends/gpu/target_config:specs/a100_sxm_80.txtpb",
    "a100": "//xla/backends/gpu/target_config:specs/a100_sxm_80.txtpb",
    "a6000": "//xla/backends/gpu/target_config:specs/a6000.txtpb",
    "b200": "//xla/backends/gpu/target_config:specs/b200.txtpb",
    "b300": "//xla/backends/gpu/target_config:specs/b300.txtpb",
    "bmg_g21": "//xla/backends/gpu/target_config:specs/bmg_g21.txtpb",
    "gb200": "//xla/backends/gpu/target_config:specs/gb200.txtpb",
    "gb300": "//xla/backends/gpu/target_config:specs/gb300.txtpb",
    "gfx1250": "//xla/backends/gpu/target_config:specs/gfx1250.txtpb",
    "h100_pcie": "//xla/backends/gpu/target_config:specs/h100_pcie.txtpb",
    "h100_sxm": "//xla/backends/gpu/target_config:specs/h100_sxm.txtpb",
    "h100": "//xla/backends/gpu/target_config:specs/h100_sxm.txtpb",
    "h200": "//xla/backends/gpu/target_config:specs/h200.txtpb",
    "mi200": "//xla/backends/gpu/target_config:specs/mi200.txtpb",
    "p100": "//xla/backends/gpu/target_config:specs/p100.txtpb",
    "pvc": "//xla/backends/gpu/target_config:specs/pvc.txtpb",
    "rtx6000pro": "//xla/backends/gpu/target_config:specs/rtx6000pro.txtpb",
    "v100": "//xla/backends/gpu/target_config:specs/v100.txtpb",
}

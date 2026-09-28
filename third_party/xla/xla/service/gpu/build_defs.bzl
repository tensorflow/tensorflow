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
# ==============================================================================

""" GPU-specific build macros.
"""

load("//xla/tsl:package_groups.bzl", "DEFAULT_LOAD_VISIBILITY")

visibility(DEFAULT_LOAD_VISIBILITY)

def get_cub_sort_kernel_types(name = ""):
    """ List of supported types for CUB sort kernels.
    """
    return [
        "bf16",
        "f16",
        "f32",
        "f64",
        "s8",
        "s16",
        "s32",
        "s64",
        "u8",
        "u16",
        "u32",
        "u64",
        "u16_b16",
        "u16_b32",
        "u16_b64",
        "u32_b16",
        "s32_b16",
        "s32_b32",
        "s32_b64",
        "u32_b32",
        "u32_b64",
        "u64_b16",
        "u64_b32",
        "u64_b64",
        "u8_b16",
        "u8_b32",
        "u8_b64",
        "f32_b16",
        "f32_b32",
        "f32_b64",
    ]

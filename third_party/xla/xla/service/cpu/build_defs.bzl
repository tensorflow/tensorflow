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

"""build_defs for service/cpu."""

load(
    "//xla/tsl:package_groups.bzl",
    "DEFAULT_LOAD_VISIBILITY",
    "LEGACY_SERVICE_CPU_BUILD_DEFS_USERS",
)
load("//xla/tsl:tsl.bzl", "clean_dep")

visibility(DEFAULT_LOAD_VISIBILITY + LEGACY_SERVICE_CPU_BUILD_DEFS_USERS)

def runtime_copts():
    """Returns copts used for CPU runtime libraries."""
    return (["-DEIGEN_AVOID_STL_ARRAY"] + select({
        clean_dep("//xla/tsl:android_arm"): ["-mfpu=neon"],
        "//conditions:default": [],
    }) + select({
        clean_dep("//xla/tsl:android"): ["-O2"],
        "//conditions:default": [],
    }))

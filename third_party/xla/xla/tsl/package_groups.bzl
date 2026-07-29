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

"""TSL package_group definitions"""

DEFAULT_LOAD_VISIBILITY = ["public"]
LEGACY_SERVICE_CPU_BUILD_DEFS_USERS = []
LEGACY_TSL_FRAMEWORK_CONTRACTION_BUILD_DEFS_USERS = []
LEGACY_TSL_PLATFORM_BUILD_CONFIG_ROOT_USERS = []
LEGACY_TSL_PLATFORM_BUILD_CONFIG_USERS = []
LEGACY_XLA_USERS = []
LEGACY_TSL_TSL_USERS = []
LEGACY_TSL_PROFILER_BUILDS_BUILD_CONFIG_USERS = []
LEGACY_TSL_PLATFORM_RULES_CC_USERS = []
LEGACY_TSL_PLATFORM_DEFAULT_BUILD_CONFIG_USERS = []

def tsl_package_groups(name = "tsl_package_groups"):
    native.package_group(
        name = "internal",
        packages = ["//..."],
    )

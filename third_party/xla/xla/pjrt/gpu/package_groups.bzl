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

"""Package groups for XLA GPU internal."""

def xla_gpu_internal_packages(name = "xla_gpu_internal_packages"):
    native.package_group(
        name = "legacy_gpu_client_users",
        packages = ["//..."],
    )

    native.package_group(
        name = "legacy_gpu_topology_users",
        packages = ["//..."],
    )

    native.package_group(
        name = "legacy_gpu_internal_users",
        packages = ["//..."],
    )

    native.package_group(
        name = "legacy_se_gpu_pjrt_compiler_users",
        packages = ["//..."],
    )

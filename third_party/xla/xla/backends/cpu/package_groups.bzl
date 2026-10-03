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

"""Package groups for XLA:CPU backend internal access."""

# Integrations should use PJRT as the API to access XLA.
def xla_cpu_backend_access(name = "xla_cpu_backend_access"):
    native.package_group(
        name = "xla_backend_cpu_internal_access",
        packages = ["//..."],
    )

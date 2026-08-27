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

"""Module extension for rocm."""

load("@rules_ml_toolchain//gpu/rocm:hipcc_configure.bzl", "hipcc_configure")
load("//third_party/gpus:rocm_configure.bzl", "rocm_configure")

def _rocm_configure_ext_impl(_mctx):
    rocm_configure(
        name = "local_config_rocm",
    )
    hipcc_configure(
        name = "config_rocm_hipcc",
        rocm_dist = "@local_config_rocm//rocm:toolchain_data",
    )

rocm_configure_ext = module_extension(
    implementation = _rocm_configure_ext_impl,
)

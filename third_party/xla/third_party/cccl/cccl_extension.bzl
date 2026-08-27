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

"""CCCL extension for Bazel modules."""

load("@rules_ml_toolchain//gpu/cuda:cuda_redist_init_repositories.bzl", "cuda_redist_init_repositories")
load("//third_party/cccl:workspace.bzl", "CCCL_3_2_0_DIST_DICT", "CCCL_GITHUB_VERSIONS_TO_BUILD_TEMPLATES")

def _cccl_extension_impl(ctx):  # @unused
    cuda_redist_init_repositories(
        cuda_redistributions = CCCL_3_2_0_DIST_DICT,
        redist_versions_to_build_templates = CCCL_GITHUB_VERSIONS_TO_BUILD_TEMPLATES,
    )

cccl_extension = module_extension(
    implementation = _cccl_extension_impl,
)

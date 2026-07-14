# Copyright 2025 The TensorFlow Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Hermetic NVSHMEM repository initialization. Consult the WORKSPACE on how to use it."""

load(
    "//third_party/gpus:nvidia_common_rules.bzl",
    "get_redistribution_urls",
    "get_version_and_template_lists",
    "redist_init_repository",
)
load(
    "//third_party/gpus/cuda/hermetic:cuda_redist_versions.bzl",
    "MIRRORED_TAR_NVSHMEM_REDIST_PATH_PREFIX",
    "NVSHMEM_REDIST_PATH_PREFIX",
    "NVSHMEM_REDIST_VERSIONS_TO_BUILD_TEMPLATES",
)

def nvshmem_redist_init_repository(
        nvshmem_redistributions,
        nvshmem_redist_path_prefix = NVSHMEM_REDIST_PATH_PREFIX,
        mirrored_tar_nvshmem_redist_path_prefix = MIRRORED_TAR_NVSHMEM_REDIST_PATH_PREFIX,
        redist_versions_to_build_templates = NVSHMEM_REDIST_VERSIONS_TO_BUILD_TEMPLATES):
    # buildifier: disable=function-docstring-args
    """Initializes NVSHMEM repository."""
    if "libnvshmem" in nvshmem_redistributions.keys():
        url_dict = get_redistribution_urls(nvshmem_redistributions["libnvshmem"])
    else:
        url_dict = {}
    repo_data = redist_versions_to_build_templates["libnvshmem"]
    versions, templates = get_version_and_template_lists(
        repo_data["version_to_template"],
    )
    redist_init_repository(
        name = repo_data["repo_name"],
        versions = versions,
        build_templates = templates,
        url_dict = url_dict,
        redist_path_prefix = nvshmem_redist_path_prefix,
        mirrored_tar_redist_path_prefix = mirrored_tar_nvshmem_redist_path_prefix,
        redist_version_env_vars = ["HERMETIC_NVSHMEM_VERSION"],
        local_path_env_var = "LOCAL_NVSHMEM_PATH",
        use_tar_file_env_var = "USE_NVSHMEM_TAR_ARCHIVE_FILES",
        target_arch_env_var = "NVSHMEM_REDIST_TARGET_PLATFORM",
        local_source_dirs = ["include", "lib", "bin"],
    )

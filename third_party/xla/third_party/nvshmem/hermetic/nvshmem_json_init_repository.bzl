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

"""Hermetic NVSHMEM redistributions JSON repository initialization. Consult the WORKSPACE on how to use it."""

load(
    "//third_party/gpus:nvidia_common_rules.bzl",
    "json_init_repository",
)
load(
    "//third_party/gpus/cuda/hermetic:cuda_redist_versions.bzl",
    "MIRRORED_TARS_NVSHMEM_REDIST_JSON_DICT",
    "NVSHMEM_REDIST_JSON_DICT",
)

def nvshmem_json_init_repository(
        nvshmem_json_dict = NVSHMEM_REDIST_JSON_DICT,
        mirrored_tars_nvshmem_json_dict = MIRRORED_TARS_NVSHMEM_REDIST_JSON_DICT):
    json_init_repository(
        name = "nvshmem_redist_json",
        toolkit_name = "NVSHMEM",
        json_dict = nvshmem_json_dict,
        mirrored_tars_json_dict = mirrored_tars_nvshmem_json_dict,
        redist_version_env_vars = ["HERMETIC_NVSHMEM_VERSION"],
        local_path_env_var = "LOCAL_NVSHMEM_PATH",
        use_tar_file_env_var = "USE_NVSHMEM_TAR_ARCHIVE_FILES",
    )

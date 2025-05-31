# Copyright 2024 The TensorFlow Authors. All rights reserved.
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

"""Hermetic CUDA redistributions JSON repository initialization. Consult the WORKSPACE on how to use it."""

load(
    "//third_party/gpus:nvidia_common_rules.bzl",
    "json_init_repository",
)
load(
    "//third_party/gpus/cuda/hermetic:cuda_redist_versions.bzl",
    "CUDA_REDIST_JSON_DICT",
    "CUDNN_REDIST_JSON_DICT",
    "MIRRORED_TARS_CUDA_REDIST_JSON_DICT",
    "MIRRORED_TARS_CUDNN_REDIST_JSON_DICT",
)

def _combined_redist_json_impl(repository_ctx):
    repository_ctx.file(
        "distributions.bzl",
        """load(
    "@standalone_cuda_redist_json//:distributions.bzl",
    "CUDA_REDISTRIBUTIONS",
)
load(
    "@cudnn_redist_json//:distributions.bzl",
    "CUDNN_REDISTRIBUTIONS",
)

CUDA_REDISTRIBUTIONS = CUDA_REDISTRIBUTIONS

CUDNN_REDISTRIBUTIONS = CUDNN_REDISTRIBUTIONS
""",
    )

_combined_redist_json = repository_rule(
    implementation = _combined_redist_json_impl,
)

def cuda_json_init_repository(
        cuda_json_dict = CUDA_REDIST_JSON_DICT,
        cudnn_json_dict = CUDNN_REDIST_JSON_DICT,
        mirrored_tars_cuda_json_dict = MIRRORED_TARS_CUDA_REDIST_JSON_DICT,
        mirrored_tars_cudnn_json_dict = MIRRORED_TARS_CUDNN_REDIST_JSON_DICT):
    json_init_repository(
        name = "standalone_cuda_redist_json",
        toolkit_name = "CUDA",
        json_dict = cuda_json_dict,
        mirrored_tars_json_dict = mirrored_tars_cuda_json_dict,
        redist_version_env_vars = ["HERMETIC_CUDA_VERSION", "TF_CUDA_VERSION"],
        local_path_env_var = "LOCAL_CUDA_PATH",
        use_tar_file_env_var = "USE_CUDA_TAR_ARCHIVE_FILES",
        environ = [
            "HERMETIC_CUDA_VERSION",
            "TF_CUDA_VERSION",
            "LOCAL_CUDA_PATH",
            "USE_CUDA_TAR_ARCHIVE_FILES",
        ],
    )

    json_init_repository(
        name = "cudnn_redist_json",
        toolkit_name = "CUDNN",
        json_dict = cudnn_json_dict,
        mirrored_tars_json_dict = mirrored_tars_cudnn_json_dict,
        redist_version_env_vars = ["HERMETIC_CUDNN_VERSION", "TF_CUDNN_VERSION"],
        local_path_env_var = "LOCAL_CUDNN_PATH",
        use_tar_file_env_var = "USE_CUDA_TAR_ARCHIVE_FILES",
        environ = [
            "HERMETIC_CUDNN_VERSION",
            "TF_CUDNN_VERSION",
            "LOCAL_CUDNN_PATH",
            "USE_CUDA_TAR_ARCHIVE_FILES",
        ],
    )

    # This repository is needed to combine the CUDA and CUDNN redistributions.
    _combined_redist_json(
        name = "cuda_redist_json",
        environ = [
            "HERMETIC_CUDA_VERSION",
            "TF_CUDA_VERSION",
            "LOCAL_CUDA_PATH",
            "HERMETIC_CUDNN_VERSION",
            "TF_CUDNN_VERSION",
            "LOCAL_CUDNN_PATH",
            "USE_CUDA_TAR_ARCHIVE_FILES",
        ],
    )

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

""" Provides the CCCL headers for CUDA. """

# CCCL version 2.8.5
# https://github.com/NVIDIA/cccl/releases/tag/v2.8.5
_CCCL_2_8_5_GITHUB_ARCHIVE = {
    "full_path": "https://github.com/NVIDIA/cccl/releases/download/v2.8.5/cccl-src-v2.8.5.tar.gz",
    "sha256": "631cb620e78904fb30f4ef6e033efb7716af1934bc170dce9cbbf4cc48777ca2",
    "strip_prefix": "cccl-src-v2.8.5",
}

# CCCL version 3.2.0
# https://github.com/NVIDIA/cccl/releases/tag/v3.2.0
_CCCL_3_2_0_GITHUB_ARCHIVE = {
    "full_path": "https://github.com/NVIDIA/cccl/releases/download/v3.2.0/cccl-src-v3.2.0.tar.gz",
    "sha256": "b5cd66e240201f5a06af2a75eaffdf05a6c63829edada33ff569ada0037f8086",
    "strip_prefix": "cccl-src-v3.2.0",
}

CCCL_2_8_5_DIST_DICT = {
    "cuda_cccl": {
        "linux-x86_64": _CCCL_2_8_5_GITHUB_ARCHIVE,
        "linux-sbsa": _CCCL_2_8_5_GITHUB_ARCHIVE,
    },
}

CCCL_3_2_0_DIST_DICT = {
    "cuda_cccl": {
        "linux-x86_64": _CCCL_3_2_0_GITHUB_ARCHIVE,
        "linux-sbsa": _CCCL_3_2_0_GITHUB_ARCHIVE,
    },
}

CCCL_GITHUB_VERSIONS_TO_BUILD_TEMPLATES = {
    "cuda_cccl": {
        "repo_name": "cuda_cccl",
        "version_to_template": {
            "any": "@rules_ml_toolchain//gpu/cuda/build_templates:cuda_cccl_github.BUILD.tpl",
        },
        "local": {
            "local_path_env_var": "LOCAL_CCCL_PATH",
            "source_dirs": ["thrust", "libcudacxx", "cub"],
            "version_to_template": {
                "any": "@rules_ml_toolchain//gpu/cuda/build_templates:cuda_cccl_github.BUILD.tpl",
            },
        },
    },
}

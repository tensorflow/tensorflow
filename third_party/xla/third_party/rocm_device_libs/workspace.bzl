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

"""Provides the repository macro to import Rocm-Device-Libs"""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Imports Rocm-Device-Libs."""
    LLVM_COMMIT = "53996464fa8d94b182ac4aaa7dc3a109ab524f45"
    LLVM_SHA256 = "92f4ee3cb7de1f00e0f46f95558246ef18e5eb8ddb364fc98daad6622a08fac7"

    tf_http_archive(
        name = "rocm_device_libs",
        sha256 = LLVM_SHA256,
        strip_prefix = "llvm-project-{commit}/amd/device-libs".format(commit = LLVM_COMMIT),
        urls = tf_mirror_urls("https://github.com/ROCm/llvm-project/archive/{commit}.tar.gz".format(commit = LLVM_COMMIT)),
        build_file = "//third_party/rocm_device_libs:rocm_device_libs.BUILD",
        patch_file = [
            "//third_party/rocm_device_libs:prepare_builtins.patch",
        ],
        link_files = {
            "//third_party/rocm_device_libs:build_defs.bzl": "build_defs.bzl",
        },
    )

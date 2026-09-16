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

"""Loads NVIDIA NVTX 3 headers."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "nvtx_archive",
        build_file = "//third_party:nvtx/BUILD.bazel",
        sha256 = "f244c5eec33f9769123a755dda1e9b80339345ab278cf9542ff34677c88804b5",
        strip_prefix = "NVTX-3.5.0/c/include",
        urls = tf_mirror_urls(
            "https://github.com/NVIDIA/NVTX/archive/refs/tags/v3.5.0.tar.gz",
        ),
    )

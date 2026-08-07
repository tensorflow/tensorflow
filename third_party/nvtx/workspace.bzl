# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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

"""Loads the nvtx library, used by TF."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "nvtx_archive",
        build_file = "@xla//third_party:nvtx/BUILD.bazel",
        sha256 = "5a581c3234c5a6b2fd94363e3fdd5a4f5d2a3d9c53c4b9442b0784e6cdfe722c",
        strip_prefix = "NVTX-2942f167cc30c5e3a44a2aecd5b0d9c07ff61a07/c/include",
        urls = tf_mirror_urls("https://github.com/NVIDIA/NVTX/archive/2942f167cc30c5e3a44a2aecd5b0d9c07ff61a07.tar.gz"),
    )

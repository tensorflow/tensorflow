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

"""Loads the nccl library, used by TF."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "nccl_archive",
        build_file = "@xla//third_party/nccl:archive.BUILD",
        patch_file = ["@xla//third_party/nccl:archive.patch"],
        sha256 = "292a7f7a27b6754acaf46b5506a60758ca7b18cc1dfbd3d1d4e1e229d0863b4e",
        strip_prefix = "nccl-2.30.7-1",
        urls = tf_mirror_urls("https://github.com/NVIDIA/nccl/archive/refs/tags/v2.30.7-1.tar.gz"),
    )

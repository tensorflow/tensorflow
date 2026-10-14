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

"""Loads the FlashAttention-2 CUDA kernels, used by the FlashAttention op."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "flash_attention",
        build_file = "//third_party/flash_attention:flash_attention.BUILD",
        patch_file = ["//third_party/flash_attention:flash_attention.patch"],
        sha256 = "61cd5e91507ad7f04dc7c207d8bc8bfb1e43b56b806e51febbc27faeaee208ba",
        strip_prefix = "flash-attention-2.8.3/csrc/flash_attn",
        urls = tf_mirror_urls("https://github.com/Dao-AILab/flash-attention/archive/refs/tags/v2.8.3.tar.gz"),
    )

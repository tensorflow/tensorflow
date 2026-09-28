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

"""Loads the llvm_openmp library, used by TF."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "llvm_openmp",
        build_file = "@xla//third_party/llvm_openmp:BUILD.bazel",
        patch_file = ["@xla//third_party/llvm_openmp:openmp_switch_default_patch.patch"],
        sha256 = "d19f728c8e04fb1e94566c8d76aef50ec926cd2f95ef3bf1e0a5de4909b28b44",
        strip_prefix = "openmp-10.0.1.src",
        urls = tf_mirror_urls("https://github.com/llvm/llvm-project/releases/download/llvmorg-10.0.1/openmp-10.0.1.src.tar.xz"),
    )

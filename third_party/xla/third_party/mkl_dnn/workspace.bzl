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

"""oneAPI Deep Neural Network Library (oneDNN)"""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "onednn",
        build_file = "//third_party/mkl_dnn:mkldnn_v1.BUILD",
        patch_file = ["//third_party/mkl_dnn:setting_tsan_race_v3_7_3.patch"],
        sha256 = "071f289dc961b43a3b7c8cbe8a305290a7c5d308ec4b2f586397749abdc88296",
        strip_prefix = "oneDNN-3.7.3",
        urls = tf_mirror_urls("https://github.com/oneapi-src/oneDNN/archive/refs/tags/v3.7.3.tar.gz"),
    )

    tf_http_archive(
        name = "onednn_async",
        build_file = "//third_party/mkl_dnn:mkldnn_v1_async.BUILD",
        # Rename the gtest module in oneDNN's third_party to avoid
        # conflict with Google's gtest module
        patch_file = [
            "//third_party/mkl_dnn:setting_tsan_race_v3_12_3.patch",
            "//third_party/mkl_dnn:rename_gtest.patch",
        ],
        sha256 = "12d8551668515c64adae2b26e5ebeeefa48b1ad1035717fe8718fd702c5e066b",
        strip_prefix = "oneDNN-3.12.3",
        urls = tf_mirror_urls("https://github.com/oneapi-src/oneDNN/archive/refs/tags/v3.12.3.tar.gz"),
    )

    tf_http_archive(
        name = "onednn_async_aarch64",
        build_file = "//third_party/mkl_dnn:mkldnn_v1_async_aarch64.BUILD",
        # Rename the gtest module in oneDNN's third_party to avoid
        # conflict with Google's gtest module
        patch_file = [
            "//third_party/mkl_dnn:onednn_async_aarch64_pr_5790.patch",
            "//third_party/mkl_dnn:onednn_async_aarch64_pr_5828.patch",
            "//third_party/mkl_dnn:onednn_async_aarch64_pr_6033.patch",
        ],
        sha256 = "cceec79d37474df43589eb951602f1a820eedbbb61e58021dce2faaff1da827b",
        strip_prefix = "oneDNN-80fb15f600887e22c0c7989ce6201723e6c11536",
        urls = tf_mirror_urls("https://github.com/uxlfoundation/oneDNN/archive/80fb15f600887e22c0c7989ce6201723e6c11536.tar.gz"),
    )

    tf_http_archive(
        name = "mkl_dnn_acl_compatible",
        build_file = "//third_party/mkl_dnn:mkldnn_acl.BUILD",
        patch_file = [
            "//third_party/mkl_dnn:onednn_acl_aarch64_build_fixes.patch",
            "//third_party/mkl_dnn:rename_gtest.patch",
        ],
        sha256 = "7293a85e146c2710dcf4f7257fdebb91020004cf1627c8de684b814c2498c81a",
        strip_prefix = "oneDNN-3.11.3",
        urls = tf_mirror_urls("https://github.com/oneapi-src/oneDNN/archive/refs/tags/v3.11.3.tar.gz"),
    )

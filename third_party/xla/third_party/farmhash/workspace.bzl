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

"""Provides the repository macro to import farmhash."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Imports farmhash."""

    # Attention: tools parse and update these lines.
    # LINT.IfChange
    FARMHASH_COMMIT = "0d859a811870d10f53a594927d0d0b97573ad06d"
    FARMHASH_SHA256 = "18392cf0736e1d62ecbb8d695c31496b6507859e8c75541d7ad0ba092dc52115"
    # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/farmhash.cmake)

    tf_http_archive(
        name = "farmhash_archive",
        build_file = "//third_party/farmhash:farmhash.BUILD",
        sha256 = FARMHASH_SHA256,
        strip_prefix = "farmhash-{commit}".format(commit = FARMHASH_COMMIT),
        urls = tf_mirror_urls("https://github.com/google/farmhash/archive/{commit}.tar.gz".format(commit = FARMHASH_COMMIT)),
    )

    tf_http_archive(
        name = "farmhash_gpu_archive",
        build_file = "//third_party/farmhash:farmhash_gpu.BUILD",
        patch_file = ["//third_party/farmhash:farmhash_support_cuda.patch"],
        sha256 = FARMHASH_SHA256,
        strip_prefix = "farmhash-{commit}".format(commit = FARMHASH_COMMIT),
        urls = tf_mirror_urls("https://github.com/google/farmhash/archive/{commit}.tar.gz".format(commit = FARMHASH_COMMIT)),
    )

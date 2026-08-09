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

"""Provides the repository macro to import gemmlowp."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Imports gemmlowp."""

    # Attention: tools parse and update these lines.
    # LINT.IfChange
    GEMMLOWP_COMMIT = "16e8662c34917be0065110bfcd9cc27d30f52fdf"
    GEMMLOWP_SHA256 = "7dc418717c8456473fac4ff2288b71057e3dcb72894524c734a4362cdb51fa8b"
    # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/gemmlowp.cmake)

    tf_http_archive(
        name = "gemmlowp",
        sha256 = GEMMLOWP_SHA256,
        strip_prefix = "gemmlowp-{commit}".format(commit = GEMMLOWP_COMMIT),
        urls = tf_mirror_urls("https://github.com/google/gemmlowp/archive/{commit}.zip".format(commit = GEMMLOWP_COMMIT)),
    )

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

"""Loads the ruy library, used by TensorFlow Lite."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "ruy",
        # LINT.IfChange
        sha256 = "70a3ebd5e353293a207c9d54ce9ae004a3b8b2e266fe42ca4fbc0e971e08c375",
        strip_prefix = "ruy-690c14c441387a4ea6e07a9ed89657cec8200b92",
        urls = tf_mirror_urls("https://github.com/google/ruy/archive/690c14c441387a4ea6e07a9ed89657cec8200b92.zip"),
        # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/ruy.cmake)
        build_file = "//third_party/ruy:BUILD",
    )

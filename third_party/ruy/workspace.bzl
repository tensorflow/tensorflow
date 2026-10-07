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
        sha256 = "64a001a81c05743736ed8695edd521c6093519ef2edb6635ffa142e827cfb86b",
        strip_prefix = "ruy-2af88863614a8298689cc52b1a47b3fcad7be835",
        urls = tf_mirror_urls("https://github.com/google/ruy/archive/2af88863614a8298689cc52b1a47b3fcad7be835.zip"),
        # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/ruy.cmake)
        build_file = "//third_party/ruy:BUILD",
        patch_file = ["//third_party/ruy:ruy.patch"],
    )

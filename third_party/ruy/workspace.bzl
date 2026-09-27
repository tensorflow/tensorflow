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
        sha256 = "a22c42e80c7bb450db8492728e4742ee66f46d5458c45fe67ce2c9b61240630c",
        strip_prefix = "ruy-3286a34cc8de6149ac6844107dfdffac91531e72",
        urls = tf_mirror_urls("https://github.com/google/ruy/archive/3286a34cc8de6149ac6844107dfdffac91531e72.zip"),
        # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/ruy.cmake)
        build_file = "//third_party/ruy:BUILD",
    )

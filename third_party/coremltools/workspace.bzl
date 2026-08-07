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

"""Loads the coremltools library, used by TF."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "coremltools",
        sha256 = "37d4d141718c70102f763363a8b018191882a179f4ce5291168d066a84d01c9d",
        strip_prefix = "coremltools-8.0",
        build_file = "//third_party/coremltools:coremltools.BUILD",
        urls = tf_mirror_urls("https://github.com/apple/coremltools/archive/8.0.tar.gz"),
    )

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

"""Loads the abseil-py library, used by TF."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "absl_py",
        sha256 = "8a3d0830e4eb4f66c4fa907c06edf6ce1c719ced811a12e26d9d3162f8471758",
        strip_prefix = "abseil-py-2.1.0",
        urls = tf_mirror_urls("https://github.com/abseil/abseil-py/archive/refs/tags/v2.1.0.tar.gz"),
    )

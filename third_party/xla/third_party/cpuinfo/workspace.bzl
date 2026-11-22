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

"""cpuinfo is a library to detect essential CPU features."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "cpuinfo",
        sha256 = "1cffb88e4b6cbb59eb67253cc0b1365e8d907afabd90b012299df780961d3b11",
        strip_prefix = "cpuinfo-731932997e55e84900e6f95aefb6edd167b0edcc",
        urls = tf_mirror_urls("https://github.com/pytorch/cpuinfo/archive/731932997e55e84900e6f95aefb6edd167b0edcc.zip"),
    )

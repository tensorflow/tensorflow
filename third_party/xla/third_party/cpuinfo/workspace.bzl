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
        sha256 = "60f758240032413a31d288e7c76e8a3a81e869c8d6ce05869ad799b33e7b0a63",
        strip_prefix = "cpuinfo-783a616b2197d7e48d6507210d1d0009fbf1f590",
        urls = tf_mirror_urls("https://github.com/pytorch/cpuinfo/archive/783a616b2197d7e48d6507210d1d0009fbf1f590.zip"),
    )

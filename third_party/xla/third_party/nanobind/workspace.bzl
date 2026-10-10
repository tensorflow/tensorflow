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

"""Loads the nanobind library."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "nanobind",
        strip_prefix = "nanobind-da42c8a5890eff09e672e7cfd953627655aa3e28",
        sha256 = "3b24ad20ce0429b1a501151eeaeff723451b9e9b812887b7f9cb8575fa5e7ec7",
        urls = tf_mirror_urls("https://github.com/wjakob/nanobind/archive/da42c8a5890eff09e672e7cfd953627655aa3e28.tar.gz"),
        build_file = "//third_party/nanobind:nanobind.BUILD",
    )

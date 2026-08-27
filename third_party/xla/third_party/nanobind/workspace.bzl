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
        strip_prefix = "nanobind-e2dc00f7a34f935c6cf91948776d59c4709e9fe6",
        sha256 = "99fea0ea1c61b94a02811f7ad4915e70145b8acdb4b65bb67a4e56981d1f7d32",
        urls = tf_mirror_urls("https://github.com/wjakob/nanobind/archive/e2dc00f7a34f935c6cf91948776d59c4709e9fe6.tar.gz"),
        build_file = "//third_party/nanobind:nanobind.BUILD",
    )

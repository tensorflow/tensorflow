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
        strip_prefix = "nanobind-db4827f06f6f1680e5d4004c95fc8d69299dba8b",
        sha256 = "60c350b2d64cdf2c1d8e46433d38dba0fb69ba118f9077c1c7e74228e6dec1e0",
        urls = tf_mirror_urls("https://github.com/wjakob/nanobind/archive/db4827f06f6f1680e5d4004c95fc8d69299dba8b.tar.gz"),
        build_file = "//third_party/nanobind:nanobind.BUILD",
    )

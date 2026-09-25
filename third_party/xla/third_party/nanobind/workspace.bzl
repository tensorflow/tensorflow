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
        strip_prefix = "nanobind-68480a9e6883bb1bf65e1925423321e2a89d07f9",
        sha256 = "177ca257aacbf58004809f8cd1c20ad0416d8ace0a884c16a3c03ea87e81876b",
        urls = tf_mirror_urls("https://github.com/wjakob/nanobind/archive/68480a9e6883bb1bf65e1925423321e2a89d07f9.tar.gz"),
        build_file = "//third_party/nanobind:nanobind.BUILD",
    )

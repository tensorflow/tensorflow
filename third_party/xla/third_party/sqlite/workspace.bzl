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

"""Provides the repository macro to import sqlite3.

The non-standard name used aligns with TensorFlow workspace definition, to
satisfy TF presubmits.
"""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "org_sqlite",
        build_file = "//third_party/sqlite:BUILD",
        sha256 = "8a310d0a16c7a90cacd4c884e70faa51c902afed2a89f63aaa0126ab83558a32",
        strip_prefix = "sqlite-amalgamation-3530200",
        system_build_file = "//third_party/systemlibs:sqlite.BUILD",
        urls = tf_mirror_urls("https://sqlite.org/2026/sqlite-amalgamation-3530200.zip"),
    )

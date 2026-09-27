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

"""Loads the sqlite library, used by TF."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "org_sqlite",
        build_file = "//third_party/sqlite:sqlite.BUILD",
        sha256 = "9ad6d16cbc1df7cd55c8b55127c82a9bca5e9f287818de6dc87e04e73599d754",
        strip_prefix = "sqlite-amalgamation-3500300",
        system_build_file = "//third_party/systemlibs:sqlite.BUILD",
        urls = tf_mirror_urls("https://www.sqlite.org/2025/sqlite-amalgamation-3500300.zip"),
    )

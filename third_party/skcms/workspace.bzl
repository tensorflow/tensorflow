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

"""Point to the skcms repo on GitHub."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "skcms",
        strip_prefix = "skcms-2025_09_16",
        sha256 = "08c45dff8ede1b56a6e7d1e9fdaf113dd91ab28ddcbcf696229b683e5e9af45a",
        urls = tf_mirror_urls("https://github.com/google/skcms/archive/refs/tags/2025_09_16.tar.gz"),
        build_file = "//third_party/skcms:skcms.BUILD.bazel",
    )

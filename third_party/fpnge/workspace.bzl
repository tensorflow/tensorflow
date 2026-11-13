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

"""Point to the fpnge repo on GitHub."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    FPNGE_COMMIT = "097ebccf3d3db443d9a8f238b1deed010a82f961"
    tf_http_archive(
        name = "fpnge",
        strip_prefix = "fpnge-" + FPNGE_COMMIT,
        sha256 = "bfbed770bd2e036ab601dc2cca2e040a9553d8d06f06ad7732824823db94db85",
        urls = tf_mirror_urls("https://github.com/veluca93/fpnge/archive/" + FPNGE_COMMIT + ".tar.gz"),
        build_file = "//third_party/fpnge:fpnge.BUILD.bazel",
    )

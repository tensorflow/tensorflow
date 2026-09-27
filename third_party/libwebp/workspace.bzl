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

"""Point to the libwebp repo on GitHub."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    # Use the same libwebp release as tensorstore
    tf_http_archive(
        name = "libwebp",
        strip_prefix = "libwebp-1.4.0",
        sha256 = "12af50c45530f0a292d39a88d952637e43fb2d4ab1883c44ae729840f7273381",
        urls = tf_mirror_urls("https://github.com/webmproject/libwebp/archive/v1.4.0.tar.gz"),
        build_file = "//third_party/libwebp:libwebp.BUILD.bazel",
    )

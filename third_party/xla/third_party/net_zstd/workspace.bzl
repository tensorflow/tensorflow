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

"""loads the net_zstd library."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "net_zstd",
        build_file = "//third_party:net_zstd.BUILD",
        sha256 = "7897bc5d620580d9b7cd3539c44b59d78f3657d33663fe97a145e07b4ebd69a4",
        strip_prefix = "zstd-1.5.7/lib",
        urls = tf_mirror_urls("https://github.com/facebook/zstd/archive/v1.5.7.zip"),  # 2025-05-20
    )

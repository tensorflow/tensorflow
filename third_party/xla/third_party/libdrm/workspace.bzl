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

"""Loads libdrm headers for ROCm compatibility."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Import libdrm headers."""

    # libdrm 2.4.120 - a recent stable version
    tf_http_archive(
        name = "libdrm",
        build_file = str(Label("//third_party/libdrm:libdrm.BUILD")),
        sha256 = "3bf55363f76c7250946441ab51d3a6cc0ae518055c0ff017324ab76cdefb327a",
        strip_prefix = "libdrm-2.4.120",
        urls = tf_mirror_urls(
            "https://dri.freedesktop.org/libdrm/libdrm-2.4.120.tar.xz",
        ),
    )

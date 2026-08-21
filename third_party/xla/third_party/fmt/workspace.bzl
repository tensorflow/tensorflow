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

"""Provides the repository macro to import fmt."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Imports fmt."""

    FMT_VERSION = "8.1.1"
    FMT_SHA256 = "3d794d3cf67633b34b2771eb9f073bde87e846e0d395d254df7b211ef1ec7346"

    tf_http_archive(
        name = "fmt",
        sha256 = FMT_SHA256,
        strip_prefix = "fmt-{version}".format(version = FMT_VERSION),
        urls = tf_mirror_urls("https://github.com/fmtlib/fmt/archive/refs/tags/{version}.tar.gz".format(version = FMT_VERSION)),
        build_file = "//third_party/fmt:fmt.BUILD",
    )

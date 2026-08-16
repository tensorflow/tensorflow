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

"""Provides the repository macro to import gutil."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Imports gutil."""

    # Attention: tools parse and update these lines.
    GUTIL_COMMIT = "b498c8d364ac96c32194f71f8f719707a398e82b"  # LTS 20250502.0
    GUTIL_SHA256 = "aeca39e4a50f9607437731aba79189a64ff51b742c00f8b80049686e7600e09f"

    tf_http_archive(
        name = "com_google_gutil",
        sha256 = GUTIL_SHA256,
        strip_prefix = "gutil-{commit}".format(commit = GUTIL_COMMIT),
        urls = tf_mirror_urls("https://github.com/google/gutil/archive/{commit}.tar.gz".format(commit = GUTIL_COMMIT)),
    )

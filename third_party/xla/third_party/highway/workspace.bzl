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
    """Imports highway."""

    HIGHWAY_VERSION = "1.3.0"
    HIGHWAY_SHA256 = "07b3c1ba2c1096878a85a31a5b9b3757427af963b1141ca904db2f9f4afe0bc2"

    tf_http_archive(
        name = "com_google_highway",
        strip_prefix = "highway-{version}".format(version = HIGHWAY_VERSION),
        sha256 = HIGHWAY_SHA256,
        urls = tf_mirror_urls("https://github.com/google/highway/archive/refs/tags/{version}.tar.gz".format(version = HIGHWAY_VERSION)),
    )

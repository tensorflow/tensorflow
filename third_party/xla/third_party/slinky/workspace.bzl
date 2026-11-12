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

"""slinky is a lightweight runtime for semi-automatical optimization of data flow pipelines for locality."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "slinky",
        sha256 = "53f6e08c00a0aeaf5563806d58587e0508a0c18039cb6bbccdede06ce2da3446",
        strip_prefix = "slinky-fba43def0e0836533d16a0500996cd76e27ca0ca",
        urls = tf_mirror_urls("https://github.com/dsharlet/slinky/archive/fba43def0e0836533d16a0500996cd76e27ca0ca.zip"),
    )

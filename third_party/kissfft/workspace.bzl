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

"""Loads the kissfft library, used by TF Lite."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "kissfft",
        strip_prefix = "kissfft-131.1.0",
        sha256 = "76c1aac87ddb7258f34b08a13f0eebf9e53afa299857568346aa5c82bcafaf1a",
        urls = tf_mirror_urls("https://github.com/mborgerding/kissfft/archive/refs/tags/131.1.0.tar.gz"),
        build_file = "//third_party/kissfft:kissfft.BUILD",
    )

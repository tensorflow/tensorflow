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

"""Point to the Highway repo on GitHub."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "highway",
        strip_prefix = "highway-1.4.0",
        sha256 = "e72241ac9524bb653ae52ced768b508045d4438726a303f10181a38f764a453c",
        urls = tf_mirror_urls("https://github.com/google/highway/archive/refs/tags/1.4.0.tar.gz"),
        patch_file = ["//third_party/com_google_highway:build.patch"],
        repo_mapping = {
            "@com_google_highway": "@highway",
        },
    )

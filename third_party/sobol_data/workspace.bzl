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

"""Loads the sobol_data library, used by TF."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "sobol_data",
        urls = tf_mirror_urls("https://github.com/joe-kuo/sobol_data/archive/835a7d7b1ee3bc83e575e302a985c66ec4b65249.tar.gz"),
        sha256 = "583d7b975e506c076fc579d9139530596906b9195b203d42361417e9aad79b73",
        strip_prefix = "sobol_data-835a7d7b1ee3bc83e575e302a985c66ec4b65249",
        build_file = "//third_party/sobol_data:sobol_data.BUILD",
    )

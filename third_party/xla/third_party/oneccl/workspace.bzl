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

"""OneAPI Collective Communication Library (oneCCL)"""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo_v1():
    tf_http_archive(
        name = "oneccl_v1",
        build_file = "//third_party/oneccl:oneccl_v1.BUILD",
        patch_file = [
            "//third_party/oneccl:ze_loader.patch",
        ],
        sha256 = "016b190557c3a5ee585fe38ce3bf8d6a0c99d7b1a55272083db455b2eff92013",
        strip_prefix = "oneCCL-4ceafd15c03ce46f11eeaf91781a92afebd3cecf",
        urls = tf_mirror_urls("https://github.com/uxlfoundation/oneCCL/archive/4ceafd15c03ce46f11eeaf91781a92afebd3cecf.tar.gz"),
    )

def repo_v2():
    tf_http_archive(
        name = "oneccl",
        build_file = "//third_party/oneccl:oneccl_v2.BUILD",
        patch_file = [
            "//third_party/oneccl:load_plugin.patch",
        ],
        sha256 = "de801277f23242d128fa4edd3e33224a450ca438ec545511fe65efa6d1426530",
        strip_prefix = "oneCCL-2022.0.0",
        urls = tf_mirror_urls("https://github.com/uxlfoundation/oneCCL/archive/refs/tags/2022.0.0.tar.gz"),
    )

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

"""KleidiAI library."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "KleidiAI",
        sha256 = "a424e81535ebb6d0117e9282fba45d70aee0bedb481f603df55e1805bed0e21b",
        strip_prefix = "kleidiai-5147983e14600ce7f8a6583dc04ba07753ac54c4",
        urls = tf_mirror_urls("https://gitlab.arm.com/kleidi/kleidiai/-/archive/5147983e14600ce7f8a6583dc04ba07753ac54c4/kleidiai-5147983e14600ce7f8a6583dc04ba07753ac54c4.zip"),
    )

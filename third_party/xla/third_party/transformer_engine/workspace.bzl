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

"""Loads the TransformerEngine library."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "transformer_engine",
        strip_prefix = "TransformerEngine-2.5",
        sha256 = "ee52ee9e43e44edc8598bc3d111eedc2445c9ebfe78a1fcab6f5c4c887020b72",
        urls = tf_mirror_urls("https://github.com/NVIDIA/TransformerEngine/archive/refs/tags/v2.5.tar.gz"),
        build_file = "//third_party/transformer_engine:transformer_engine.BUILD",
        patch_file = ["//third_party/transformer_engine:transformer_engine.patch"],
    )

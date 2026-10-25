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

"""Provides the repo macro to import brotli"""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "org_brotli",
        sha256 = "ae0fe7f40f6057cb7c592891c0de6657aaa82a1d09317150ff7eb0f662c5f5d8",
        strip_prefix = "brotli-7b93874da36fe4420a18fb105321e8c27d5ef078",
        patch_file = [
            # On a version upgrade, this patch can be regenerated with the command:
            # build_tools/dependencies/gen_disable_layering_check_patch.sh.  \
            #   https://github.com/google/brotli/archive/7b93874da36fe4420a18fb105321e8c27d5ef078.tar.gz \
            #   > third_party/brotli/layering_check.patch
            "//third_party/brotli:layering_check.patch",
        ],
        urls = tf_mirror_urls("https://github.com/google/brotli/archive/7b93874da36fe4420a18fb105321e8c27d5ef078.tar.gz"),
    )

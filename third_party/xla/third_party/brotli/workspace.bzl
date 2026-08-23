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
        sha256 = "e720a6ca29428b803f4ad165371771f5398faba397edf6778837a18599ea13ff",
        strip_prefix = "brotli-1.1.0",
        patch_file = [
            # On a version upgrade, this patch can be regenerated with the command:
            # build_tools/dependencies/gen_disable_layering_check_patch.sh.  \
            #   https://github.com/google/brotli/archive/refs/tags/v1.1.0.tar.gz \
            #   > third_party/brotli/layering_check.patch
            "//third_party/brotli:layering_check.patch",
        ],
        urls = tf_mirror_urls("https://github.com/google/brotli/archive/refs/tags/v1.1.0.tar.gz"),
    )

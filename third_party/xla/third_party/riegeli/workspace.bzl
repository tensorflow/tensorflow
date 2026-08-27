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

"""Provides the repo macro to import riegeli"""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "riegeli",
        sha256 = "f63337f63f794ba9dc7dd281b20af3d036dfe0c1a5a4b7b8dc20b39f7e323b97",
        strip_prefix = "riegeli-9f2744dc23e81d84c02f6f51244e9e9bb9802d57",
        patch_file = [
            # On a version upgrade, this patch can be regenerated with the command:
            # build_tools/dependencies/gen_disable_layering_check_patch.sh.  \
            #   https://github.com/google/riegeli/archive/9f2744dc23e81d84c02f6f51244e9e9bb9802d57.tar.gz \
            #   > third_party/riegeli/layering_check.patch
            "//third_party/riegeli:layering_check.patch",
        ],
        urls = tf_mirror_urls("https://github.com/google/riegeli/archive/9f2744dc23e81d84c02f6f51244e9e9bb9802d57.tar.gz"),
    )

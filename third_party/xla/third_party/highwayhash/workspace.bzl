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

"""loads the highwayhash library, used by TF."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "highwayhash",
        urls = tf_mirror_urls("https://github.com/google/highwayhash/archive/c13d28517a4db259d738ea4886b1f00352a3cc33.tar.gz"),
        sha256 = "c0e2b9931fbcce3bfbcd7999c3c114f404ac0f8b89775a5bbccbcaa501868e58",
        strip_prefix = "highwayhash-c13d28517a4db259d738ea4886b1f00352a3cc33",
        build_file = "//third_party/highwayhash:highwayhash.BUILD",
    )

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
        sha256 = "6b3e6630be314a28f6ea28fed14f7109b0b7c472f1e06d2dba17ffccda3b9466",
        strip_prefix = "kleidiai-dce86647385ab2638aa5abebcb652f3e4271970d",
        urls = tf_mirror_urls("https://gitlab.arm.com/kleidi/kleidiai/-/archive/dce86647385ab2638aa5abebcb652f3e4271970d/kleidiai-dce86647385ab2638aa5abebcb652f3e4271970d.zip"),
    )

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
        sha256 = "0119fa3e386c27a10c8b4bb5673eec1c4e32a1c09943e8d84cdda2c6660fb79c",
        strip_prefix = "kleidiai-0ec849c4ee7eecc8e79e6808d20bcdd10e8d137f",
        urls = tf_mirror_urls("https://gitlab.arm.com/kleidi/kleidiai/-/archive/0ec849c4ee7eecc8e79e6808d20bcdd10e8d137f/kleidiai-0ec849c4ee7eecc8e79e6808d20bcdd10e8d137f.zip"),
    )

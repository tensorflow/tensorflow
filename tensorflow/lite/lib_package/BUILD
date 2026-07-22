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
# ==============================================================================

package(default_visibility = ["//visibility:private"])

# Create the LICENSE file for libraries that are used by TensorFlow Lite
# C library.
genrule(
    name = "clicenses_generate",
    srcs = [
        # copybara:uncomment "//third_party/eigen3:LICENSE",
        "@arm_neon_2_x86_sse//:LICENSE",
        "@farmhash_archive//:COPYING",
        "@gemmlowp//:LICENSE",
    ],
    outs = ["LICENSE"],
    cmd = "$(location :concat_licenses.sh) $(SRCS) >$@",
    tools = [":concat_licenses.sh"],
)

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

# Description:
#   NEON2SSE - a header file redefining ARM Neon intrinsics in terms of SSE intrinsics
#              allowing neon code to compile and run on x64/x86 workstantions.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # 3-Clause BSD

exports_files([
    "LICENSE",
])

cc_library(
    name = "arm_neon_2_x86_sse",
    hdrs = ["NEON_2_SSE.h"],
)

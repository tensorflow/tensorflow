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

licenses(["notice"])  # 3-Clause BSD

exports_files(["LICENSE.MIT"])

cc_library(
    name = "nlohmann_json_lib",
    hdrs = glob([
        "include/nlohmann/**/*.hpp",
    ]),
    copts = [
        "-I external/nlohmann_json_lib",
    ],
    visibility = ["//visibility:public"],
    alwayslink = 1,
)

cc_library(
    name = "nlohmann_json",
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = ["nlohmann_json_lib"],
)

# For compatibility in bzlmod.
alias(
    name = "json",
    actual = ":nlohmann_json",
    visibility = ["//visibility:public"],
)

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

load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "hash",
    linkopts = ["-labsl_hash"],
    deps = [
        ":city",
        ":low_level_hash",
        "//absl/base:endian",
        "//absl/container:fixed_array",
        "//absl/numeric:int128",
        "//absl/strings",
        "//absl/types:optional",
        "//absl/types:variant",
        "//absl/utility",
    ],
)

cc_library(
    name = "city",
    linkopts = ["-labsl_city"],
    deps = [
        "//absl/base:endian",
    ],
)

cc_library(
    name = "low_level_hash",
    linkopts = ["-labsl_low_level_hash"],
    visibility = ["//visibility:private"],
    deps = [
        "//absl/base:endian",
        "//absl/numeric:int128",
    ],
)

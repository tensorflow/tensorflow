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

load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_visibility = ["//visibility:public"],
    features = ["header_modules"],
)

licenses(["notice"])

config_setting(
    name = "x86_64",
    constraint_values = [
        "@platforms//cpu:x86_64",
    ],
)

cc_library(
    name = "zstdlib",
    srcs = glob([
        "common/*.c",
        "common/*.h",
        "compress/*.c",
        "compress/*.h",
        "decompress/*.c",
        "decompress/*.h",
    ]) + select({
        ":x86_64": glob(["decompress/*_amd64.S"]),
        "//conditions:default": [],
    }),
    hdrs = glob([
        "*.h",
    ]),
    includes = ["."],
)

alias(
    name = "zstd",
    actual = ":zstdlib",
)

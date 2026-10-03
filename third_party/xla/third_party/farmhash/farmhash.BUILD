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

licenses(["notice"])  # MIT

exports_files(["COPYING"])

config_setting(
    name = "windows",
    constraint_values = [
        "@platforms//os:windows",
        "@platforms//cpu:x86_64",
    ],
)

config_setting(
    name = "windows_x86_64_clang",
    constraint_values = [
        "@platforms//os:windows",
        "@platforms//cpu:x86_64",
        "@bazel_tools//tools/cpp:clang-cl",
    ],
)

cc_library(
    name = "farmhash",
    srcs = ["src/farmhash.cc"],
    hdrs = ["src/farmhash.h"],
    # Disable __builtin_expect support on Windows
    copts = select({
        ":windows_x86_64_clang": ["-DFARMHASH_OPTIONAL_BUILTIN_EXPECT"],
        ":windows": ["/DFARMHASH_OPTIONAL_BUILTIN_EXPECT"],
        "//conditions:default": [],
    }),
    includes = ["src/."],
    visibility = ["//visibility:public"],
)

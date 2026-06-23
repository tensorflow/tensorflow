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

load("@local_config_sycl//sycl:build_defs.bzl", "sycl_library")

sycl_library(
    name = "oneccl",
    srcs = glob([
        "src/api.cpp",
        "src/debug.cpp",
    ]),
    hdrs = glob([
        "include/**/*.h",
        "src/**/*.h",
    ]),
    includes = [
        "include",
        "src",
    ],
    linkopts = ["-ldl"],
    visibility = ["//visibility:public"],
    deps = [
        "@oneccl_v1",
    ],
    alwayslink = True,
)

sycl_library(
    name = "ccl_legacy",
    srcs = glob([
        "plugins/legacy/ccl_legacy.cpp",
    ]),
    hdrs = glob([
        "src/**/*.hpp",
    ]),
    includes = [
        "include",
        "src",
    ],
    deps = [
        ":oneccl",
        "@oneccl_v1",
        "@oneccl_v1//:mpi",
    ],
    alwayslink = True,
)

sycl_library(
    name = "libs",
    visibility = ["//visibility:public"],
    deps = [
        ":ccl_legacy",
        ":oneccl",
    ],
    alwayslink = True,
)

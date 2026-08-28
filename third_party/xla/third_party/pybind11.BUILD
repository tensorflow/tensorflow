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

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "pybind11",
    hdrs = glob(
        include = [
            "include/pybind11/*.h",
            "include/pybind11/detail/*.h",
        ],
        exclude = [
            "include/pybind11/common.h",
            "include/pybind11/eigen.h",
        ],
    ),
    copts = select({
        ":msvc_compiler": [],
        "//conditions:default": [
            "-fexceptions",
            "-Wno-undefined-inline",
            "-Wno-pragma-once-outside-header",
        ],
    }),
    includes = ["include"],
    strip_include_prefix = "include",
    deps = [
        "@xla//third_party/python_runtime:headers",
    ],
)

# Used when one also needs eigen types.
cc_library(
    name = "pybind11_eigen",
    hdrs = glob(
        include = [
            "include/pybind11/*.h",
            "include/pybind11/detail/*.h",
            "include/pybind11/eigen/*.h",
        ],
        exclude = [
            "include/pybind11/common.h",
        ],
    ),
    copts = select({
        ":msvc_compiler": [],
        "//conditions:default": [
            "-fexceptions",
            "-Wno-undefined-inline",
            "-Wno-pragma-once-outside-header",
        ],
    }),
    includes = ["include"],
    strip_include_prefix = "include",
    deps = [
        "@eigen_archive//:eigen3",
        "@xla//third_party/python_runtime:headers",
    ],
)

# Needed by pybind11_bazel.
config_setting(
    name = "msvc_compiler",
    flag_values = {"@bazel_tools//tools/cpp:compiler": "msvc-cl"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "osx",
    constraint_values = ["@platforms//os:osx"],
)

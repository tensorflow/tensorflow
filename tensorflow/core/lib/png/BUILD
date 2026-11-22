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

load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load(
    "//tensorflow/core/platform:rules_cc.bzl",
    "cc_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow:__subpackages__",
    ],
    licenses = ["notice"],
)

filegroup(
    name = "hdrs",
    srcs = [
        "png_io.h",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "fast_png",
    srcs = ["fast_png.cc"],
    hdrs = ["fast_png.h"],
    copts = select({
        "//tensorflow:linux_x86_64_no_sse": [],
        "//tensorflow:linux_x86_64": ["-DTF_PNG_ENABLE_FPNGE"],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:private"],
    deps = [
        "//tensorflow/core/platform:tstring",
    ] + select({
        "//tensorflow:linux_x86_64_no_sse": [],
        "//tensorflow:linux_x86_64": [
            # Copybara only rewrites one label per line.
            "@fpnge",
        ],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "png_io",
    srcs = ["png_io.cc"],
    hdrs = ["png_io.h"],
    features = ["-layering_check"],
    visibility = ["//visibility:public"],
    deps = [
        ":fast_png",
        "//tensorflow/core/platform:byte_order",
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/platform:png",
        "//tensorflow/core/platform:stringpiece",
        "//tensorflow/core/platform:types",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings:string_view",
        "@png",
        "@zlib",
    ],
)

tf_cc_test(
    name = "fast_png_test",
    srcs = ["fast_png_test.cc"],
    extra_copts = select({
        "//tensorflow:linux_x86_64_no_sse": [],
        "//tensorflow:linux_x86_64": ["-DTF_PNG_ENABLE_FPNGE"],
        "//conditions:default": [],
    }),
    deps = [
        ":fast_png",
        ":png_io",
        "//tensorflow/core/platform:tstring",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

alias(
    name = "testdata",
    actual = "//tensorflow/core/lib/png/testdata:png_testdata",
    visibility = ["//tensorflow/core:__pkg__"],
)

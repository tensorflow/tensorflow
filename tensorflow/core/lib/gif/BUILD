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
# gif io package.

load(
    "//tensorflow:tensorflow.bzl",
    "if_android",
    "if_mobile",
    "tf_cc_test",
    "tf_copts",
)
load(
    "//tensorflow/core/platform:rules_cc.bzl",
    "cc_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow/core:__pkg__"],
    licenses = ["notice"],
)

filegroup(
    name = "hdrs",
    srcs = ["gif_io.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "gif_internal",
    srcs = [
        "gif_io.cc",
    ],
    hdrs = ["gif_io.h"],
    copts = tf_copts(),
    linkopts = select({
        "//tensorflow:freebsd": [],
        "//tensorflow:windows": [],
        "//conditions:default": ["-ldl"],
    }),
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/platform:gif",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@gif",
    ],
)

cc_library(
    name = "portable_gif_internal",
    srcs = if_mobile([
        "gif_io.cc",
        "//tensorflow/core/platform:gif_hdrs",
    ]),
    hdrs = [
        "gif_io.h",
        "//tensorflow/core/lib/core:legacy_lib_core_stringpiece_header",
        "//tensorflow/core/lib/gtl:legacy_android_gif_internal_headers",
        "//tensorflow/core/platform:gif_internal_hdrs",
        "@tsl//tsl/platform:gif_internal_hdrs",
    ],
    copts = tf_copts(),
    features = ["-layering_check"],
    linkopts = if_android(["-ldl"]),
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:byte_order",
        "//tensorflow/core/platform:gif",
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/platform:stringpiece",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:dynamic_annotations",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@eigen_archive//:eigen3",
        "@gif",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:tstring",
        "@tsl//tsl/platform:types",
        "@xla//xla/tsl/platform:types",
    ],
)

tf_cc_test(
    name = "lib_gif_io_test",
    srcs = ["gif_io_test.cc"],
    data = ["//tensorflow/core/lib/gif/testdata:gif_testdata"],
    deps = [
        ":gif_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/lib/png:png_io",
        "@com_google_absl//absl/base",
    ],
)

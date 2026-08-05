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

load("//tensorflow:tensorflow.bzl", "if_google", "tf_cc_test")
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
    srcs = ["webp_io.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "webp_io",
    srcs = ["webp_io.cc"],
    hdrs = ["webp_io.h"],
    features = ["-layering_check"],
    deps = [
        "//tensorflow/core/platform:types",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@libwebp//:webp",
    ] + if_google([
        "@libwebp//:webp_demux",
    ]),
)

alias(
    name = "testdata",
    actual = "//tensorflow/core/lib/webp/testdata:webp_testdata",
    visibility = ["//tensorflow/core:__pkg__"],
)

tf_cc_test(
    name = "webp_io_benchmark",
    srcs = ["webp_io_benchmark.cc"],
    data = ["//tensorflow/core/lib/webp/testdata:webp_testdata"],
    tags = ["benchmark"],
    deps = [
        ":webp_io",
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:path",
        "//tensorflow/core/platform:resource_loader",
        "//tensorflow/core/platform:status_matchers",
        "//tensorflow/core/platform:test",
        "//tensorflow/core/platform:test_benchmark",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest",
        "@libwebp//:webp",
    ],
)

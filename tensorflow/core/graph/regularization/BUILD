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
# A TensorFlow Graph Regularization Library

load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
)
load("//tensorflow:tensorflow.default.bzl", "filegroup")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow:__subpackages__",
    ],
    licenses = ["notice"],
)

filegroup(
    name = "test_files",
    srcs = glob([
        "testdata/bert2/**",
        "testdata/bert1/**",
    ]),
)

cc_library(
    name = "util",
    srcs = ["util.cc"],
    hdrs = ["util.h"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:numbers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "//tensorflow/core/platform:protobuf",
    ],
)

tf_cc_test(
    name = "util_test",
    size = "small",
    srcs = [
        "util_test.cc",
    ],
    deps = [
        ":util",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "simple_delete",
    srcs = ["simple_delete.cc"],
    hdrs = ["simple_delete.h"],
    deps = [
        ":util",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler:op_types",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "simple_delete_test",
    size = "small",
    srcs = [
        "simple_delete_test.cc",
    ],
    data = [
        ":test_files",
    ],
    deps = [
        ":simple_delete",
        ":util",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core/framework:node_def_proto_cc",
        "//tensorflow/core/platform:path",
        "//tensorflow/core/protobuf:for_core_protos_cc",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

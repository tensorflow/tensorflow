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

# Description: Utils for Tensorflow quantization.

load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
)
load("//tensorflow:tensorflow.default.bzl", "filegroup")
load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_proto_library",
)
load(
    "//tensorflow/core/platform:rules_cc.bzl",
    "cc_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

filegroup(
    name = "mobile_srcs_only_runtime",
    srcs = [
        "uniform_quant_ops_params.cc",
    ],
)

filegroup(
    name = "mobile_hdrs_only_runtime",
    srcs = [
        "uniform_quant_ops_params.h",
    ],
)

cc_library(
    name = "uniform_quant_ops_params",
    srcs = [
        "uniform_quant_ops_params.cc",
    ],
    hdrs = [
        "uniform_quant_ops_params.h",
    ],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/algorithm:container",
    ],
)

tf_cc_test(
    name = "uniform_quant_ops_params_test",
    srcs = [
        "uniform_quant_ops_params_test.cc",
    ],
    deps = [
        ":uniform_quant_ops_attr_proto_cc",
        ":uniform_quant_ops_params",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_proto_library(
    name = "uniform_quant_ops_attr_proto",
    srcs = ["uniform_quant_ops_attr.proto"],
    make_default_target_header_only = True,
)

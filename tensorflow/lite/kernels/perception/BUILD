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
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable", "pybind_extension")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:LICENSE"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "perception_ops",
    srcs = [
        "dense_image_warp.cc",
        "max_pool_with_argmax.cc",
        "max_unpooling_2d.cc",
        "perception_ops.cc",
    ],
    hdrs = [
        "perception_ops.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/kernels:kernel_util",
        "//tensorflow/lite/kernels:padding",
        "//tensorflow/lite/kernels/internal:common",
        "//tensorflow/lite/kernels/internal:compatibility",
        "//tensorflow/lite/kernels/internal:tensor",
        "//tensorflow/lite/kernels/internal:types",
        "@flatbuffers",
    ],
)

cc_test(
    name = "perception_ops_test",
    size = "small",
    srcs = [
        "dense_image_warp_test.cc",
        "max_pool_with_argmax_test.cc",
        "max_unpooling_2d_test.cc",
    ],
    deps = [
        ":perception_ops",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/core:framework_stable",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/kernels:test_main",
        "//tensorflow/lite/kernels:test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

pybind_extension(
    name = "pywrap_perception_ops",
    srcs = [
        "perception_ops_wrapper.cc",
    ],
    hdrs = ["perception_ops.h"],
    additional_exported_symbols = ["PerceptionOpsRegisterer"],
    enable_stub_generation = True,
    link_in_framework = True,
    pytype_srcs = [
        "pywrap_perception_ops.pyi",
    ],
    deps = [
        ":perception_ops",
        "//tensorflow/lite:mutable_op_resolver",
        "//tensorflow/lite/c:common",
        "@pybind11",
    ],
)

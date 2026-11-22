# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

load("@build_bazel_rules_apple//apple:ios.bzl", "ios_unit_test")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:objc_library.bzl", "objc_library")
load("//tensorflow/lite:build_def.bzl", "CXX17_BAZEL_ONLY_COPTS")
load("//tensorflow/lite:special_rules.bzl", "tflite_ios_lab_runner")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:LICENSE"],
    default_visibility = [
        "//visibility:public",
    ],
)

licenses(["notice"])

exports_files(["coreml_delegate.h"])

objc_library(
    name = "coreml_executor",
    srcs = ["coreml_executor.mm"],
    hdrs = ["coreml_executor.h"],
    copts = CXX17_BAZEL_ONLY_COPTS,
    sdk_frameworks = [
        "CoreML",
        "Foundation",
    ],
    deps = ["@coremltools//:mlmodel_cc_proto"],
)

cc_library(
    name = "mlmodel_proto_cc",
    deps = [
        "@coremltools//:mlmodel_cc_proto",
    ],
)

objc_library(
    name = "coreml_delegate",
    srcs = ["coreml_delegate.mm"],
    hdrs = ["coreml_delegate.h"],
    copts = CXX17_BAZEL_ONLY_COPTS,
    defines = select({
        "//tensorflow/lite/delegates:tflite_debug_delegate": ["TFLITE_DEBUG_DELEGATE"],
        "//conditions:default": [],
    }),
    module_name = "TensorFlowLiteCCoreML",
    deps = [
        ":coreml_delegate_kernel",
        ":mlmodel_proto_cc",
        "//tensorflow/lite:kernel_api",
        "//tensorflow/lite:minimal_logging",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/delegates:utils",
        "//tensorflow/lite/delegates/coreml/builders:op_builder",
        "//tensorflow/lite/delegates/coreml/builders:op_validator",
        "//tensorflow/lite/delegates/coreml/builders:util",
        "//tensorflow/lite/kernels:kernel_util",
    ],
    alwayslink = 1,
)

objc_library(
    name = "coreml_delegate_kernel",
    srcs = [
        "coreml_delegate_kernel.mm",
    ],
    hdrs = [
        "coreml_delegate_kernel.h",
    ],
    copts = CXX17_BAZEL_ONLY_COPTS,
    deps = [
        ":coreml_executor",
        "//tensorflow/lite:kernel_api",
        "//tensorflow/lite:minimal_logging",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/delegates/coreml/builders:op_builder",
        "//tensorflow/lite/kernels:kernel_util",
        "//tensorflow/lite/kernels/internal:optimized_base",
        "//tensorflow/lite/kernels/internal:types",
        "//tensorflow/lite/types:half",
    ],
)

objc_library(
    name = "coreml_delegate_test_lib",
    testonly = 1,
    srcs = ["coreml_delegate_test.mm"],
    copts = CXX17_BAZEL_ONLY_COPTS,
    sdk_frameworks = ["XCTest"],
    deps = [":coreml_delegate"],
)

ios_unit_test(
    name = "coreml_delegate_test",
    testonly = 1,
    minimum_os_version = "15.0",
    runner = tflite_ios_lab_runner("IOS_LATEST"),
    tags = [
        "noasan",
        "nomsan",
        "notsan",
        "tflite_not_portable_android",
    ],
    deps = [":coreml_delegate_test_lib"],
)

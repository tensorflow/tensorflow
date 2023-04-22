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

load("//tensorflow/lite:build_def.bzl", "tflite_copts", "tflite_linkopts")
load("//tensorflow:tensorflow.bzl", "get_compatible_with_portable")
load("//tensorflow/lite:special_rules.bzl", "tflite_portable_test_suite")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "telemetry",
    srcs = ["telemetry.cc"],
    hdrs = ["telemetry.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/experimental/acceleration/configuration:configuration_fbs",
    ],
)

cc_test(
    name = "telemetry_test",
    srcs = ["telemetry_test.cc"],
    linkopts = tflite_linkopts(),
    linkstatic = 1,
    deps = [
        ":telemetry",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/experimental/acceleration/configuration:configuration_fbs",
        "//tensorflow/lite/profiling:profile_buffer",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
    ],
)

cc_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    deps = [
        "//tensorflow/lite:kernel_api",
        "//tensorflow/lite:util",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels:kernel_util",
    ],
)

cc_library(
    name = "interpreter_utils",
    srcs = ["interpreter_utils.cc"],
    hdrs = ["interpreter_utils.h"],
    copts = tflite_copts(),
    deps = [
        "//tensorflow/lite:framework",
    ],
)

cc_test(
    name = "utils_test",
    srcs = ["utils_test.cc"],
    linkopts = tflite_linkopts(),
    linkstatic = 1,
    deps = [
        ":utils",
        "//tensorflow/lite/c:common",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "interpreter_utils_test",
    size = "small",
    srcs = ["interpreter_utils_test.cc"],
    features = ["-dynamic_link_test_srcs"],  # see go/dynamic_link_test_srcs
    deps = [
        ":delegate_test_util",
        ":interpreter_utils",
        ":utils",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:kernel_api",
        "//tensorflow/lite:string",
        "//tensorflow/lite:util",
        "//tensorflow/lite:version",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/kernels:kernel_util",
        "//tensorflow/lite/kernels/internal:compatibility",
        "//tensorflow/lite/schema:schema_conversion_utils",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/testing:util",
        "//third_party/eigen3",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
    ],
)

cc_test(
    name = "delegate_test",
    size = "small",
    srcs = ["delegate_test.cc"],
    features = ["-dynamic_link_test_srcs"],  # see go/dynamic_link_test_srcs
    deps = [
        ":delegate_test_util",
        ":interpreter_utils",
        ":utils",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:kernel_api",
        "//tensorflow/lite:string",
        "//tensorflow/lite:util",
        "//tensorflow/lite:version",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/kernels:kernel_util",
        "//tensorflow/lite/kernels/internal:compatibility",
        "//tensorflow/lite/schema:schema_conversion_utils",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/testing:util",
        "//third_party/eigen3",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
    ],
)

cc_library(
    name = "delegate_test_util",
    testonly = True,
    srcs = ["delegate_test_util.cc"],
    hdrs = ["delegate_test_util.h"],
    deps = [
        ":interpreter_utils",
        ":utils",
        "//tensorflow/lite:builtin_ops",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:string",
        "//tensorflow/lite:util",
        "//tensorflow/lite:version",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/kernels:kernel_util",
        "//tensorflow/lite/kernels/internal:compatibility",
        "//tensorflow/lite/schema:schema_conversion_utils",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/testing:util",
        "//third_party/eigen3",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_library(
    name = "serialization",
    srcs = ["serialization.cc"],
    hdrs = ["serialization.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    deps = [
        "//tensorflow/lite:minimal_logging",
        "//tensorflow/lite/c:common",
        "@farmhash_archive//:farmhash",
        "@flatbuffers",
    ],
)

cc_test(
    name = "serialization_test",
    srcs = ["serialization_test.cc"],
    linkopts = tflite_linkopts(),
    linkstatic = 1,
    deps = [
        ":serialization",
        "//tensorflow/lite:util",
        "//tensorflow/lite/c:common",
        "@com_google_googletest//:gtest_main",
    ],
)

tflite_portable_test_suite()

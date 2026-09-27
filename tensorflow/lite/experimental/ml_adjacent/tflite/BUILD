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

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:LICENSE"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "tfl_tensor_ref",
    srcs = ["tfl_tensor_ref.cc"],
    hdrs = ["tfl_tensor_ref.h"],
    deps = [
        "//tensorflow/lite:array",
        "//tensorflow/lite/core/c:c_api_types",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/experimental/ml_adjacent:lib",
        "//tensorflow/lite/kernels:kernel_util",
        "//tensorflow/lite/kernels/internal:compatibility",
    ],
)

cc_test(
    name = "tfl_tensor_ref_test",
    srcs = ["tfl_tensor_ref_test.cc"],
    deps = [
        ":tfl_tensor_ref",
        "//tensorflow/lite:util",
        "//tensorflow/lite/core/c:c_api_types",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/experimental/ml_adjacent:lib",
        "//tensorflow/lite/kernels:kernel_util",
        "//tensorflow/lite/kernels:test_util",
        "//tensorflow/lite/kernels/internal:compatibility",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "extern_call",
    srcs = ["extern_call.cc"],
    hdrs = ["extern_call.h"],
    deps = [
        ":tfl_tensor_ref",
        "//tensorflow/lite/core/c:c_api_types",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/experimental/ml_adjacent:lib",
        "//tensorflow/lite/experimental/ml_adjacent/algo:crop",
        "//tensorflow/lite/experimental/ml_adjacent/algo:resize",
        "//tensorflow/lite/kernels:kernel_util",
    ],
)

cc_test(
    name = "extern_call_test",
    srcs = ["extern_call_test.cc"],
    deps = [
        ":extern_call",
        "//tensorflow/lite/core/c:c_api_types",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/kernels:test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

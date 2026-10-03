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

load("@rules_cc//cc:cc_test.bzl", "cc_test")

# Description:
#
# This package contains shim library targets for the Async C package.
load("//tensorflow/lite/core/shims:cc_library_with_tflite.bzl", "cc_library_with_tflite")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:LICENSE"],
    default_visibility = [
        "//visibility:private",
    ],
    licenses = ["notice"],
)

cc_library_with_tflite(
    name = "backend_async_kernel_interface",
    srcs = ["backend_async_kernel_interface.cc"],
    hdrs = ["backend_async_kernel_interface.h"],
    tflite_deps = [
        "//tensorflow/lite/async/c:async_kernel",
        "//tensorflow/lite/async/c:types",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/c:common",
    ],
    visibility = ["//visibility:public"],
)

cc_test(
    name = "backend_async_kernel_interface_test",
    srcs = ["backend_async_kernel_interface_test.cc"],
    deps = [
        ":backend_async_kernel_interface",
        "//tensorflow/lite/async/c:types",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core/async:async_kernel_internal",
        "//tensorflow/lite/core/async/testing:mock_async_kernel",
        "@com_google_googletest//:gtest_main",
    ],
)

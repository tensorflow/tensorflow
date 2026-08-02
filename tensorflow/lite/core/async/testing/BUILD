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

# Test utilities for TFLite async execution.
load("//tensorflow/lite/core/shims:cc_library_with_tflite.bzl", "cc_library_with_tflite")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:LICENSE"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "test_backend",
    testonly = 1,
    srcs = ["test_backend.cc"],
    hdrs = ["test_backend.h"],
    deps = [
        "//tensorflow/lite:array",
        "//tensorflow/lite:builtin_ops",
        "//tensorflow/lite/core/async:async_kernel_internal",
        "//tensorflow/lite/core/async/c:types",
        "//tensorflow/lite/core/c:c_api_types",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/delegates:utils",
    ],
)

cc_library_with_tflite(
    name = "mock_async_kernel",
    testonly = 1,
    hdrs = ["mock_async_kernel.h"],
    deprecation = "Use //tensorflow/lite/async/testing:mock_async_kernel instead.",
    tflite_deps = [
        "//tensorflow/lite/async/testing:mock_async_kernel",
    ],
)

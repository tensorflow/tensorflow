# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
load("//tensorflow/lite:build_def.bzl", "tflite_cc_library_with_c_headers_test")
load("//tensorflow/lite/core/shims:cc_library_with_tflite.bzl", "cc_library_with_tflite_with_c_headers_test")
load("//tensorflow/lite/experimental/acceleration/mini_benchmark:special_rules.bzl", "minibenchmark_visibility_allowlist")

default_visibility_group = [
    "//tensorflow/lite/experimental/acceleration/mini_benchmark:__subpackages__",
] + minibenchmark_visibility_allowlist()

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = default_visibility_group,
    licenses = ["notice"],
)

# This target runs MiniBenchmark in a separate processon Android, while it runs MiniBenchmark
# in-process on all other platforms.
cc_library_with_tflite_with_c_headers_test(
    name = "c_api",
    hdrs = ["c_api.h"],
    deps = ["//tensorflow/lite/core/experimental/acceleration/mini_benchmark/c:c_api"],
)

# This target forces MiniBenchmark to run in-process on all platforms including Android.
tflite_cc_library_with_c_headers_test(
    name = "c_api_in_process",
    hdrs = ["c_api.h"],
    deps = [
        "//tensorflow/lite/core/experimental/acceleration/mini_benchmark/c:c_api_in_process",
    ],
)

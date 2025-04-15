# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

# C API for delegate plugins.

load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load("//tensorflow/lite:build_def.bzl", "tflite_cc_library_with_c_headers_test")
load("//tensorflow/lite/core/c:special_rules.bzl", "experimental_acceleration_api_allowlist")
load(
    "//tensorflow/lite/core/experimental/acceleration/configuration/c:special_rules.bzl",
    "delegate_plugin_visibility_allowlist",
    "gpu_plugin_visibility_allowlist",
    "xnnpack_plugin_visibility_allowlist",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow/lite:__subpackages__",
    ] + experimental_acceleration_api_allowlist(),
    licenses = ["notice"],
)

tflite_cc_library_with_c_headers_test(
    name = "delegate_plugin",
    hdrs = ["delegate_plugin.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = delegate_plugin_visibility_allowlist(),
    deps = [
        "//tensorflow/lite/core/acceleration/configuration/c:delegate_plugin",
    ],
)

tflite_cc_library_with_c_headers_test(
    name = "gpu_plugin",
    hdrs = ["gpu_plugin.h"],
    visibility = gpu_plugin_visibility_allowlist(),
    deps = [
        "//tensorflow/lite/core/acceleration/configuration/c:gpu_plugin",
    ],
)

tflite_cc_library_with_c_headers_test(
    name = "nnapi_plugin",
    hdrs = ["nnapi_plugin.h"],
    deps = [
        "//tensorflow/lite/core/acceleration/configuration/c:nnapi_plugin",
    ],
)

tflite_cc_library_with_c_headers_test(
    name = "xnnpack_plugin",
    hdrs = ["xnnpack_plugin.h"],
    visibility = xnnpack_plugin_visibility_allowlist(),
    deps = [
        "//tensorflow/lite/core/acceleration/configuration/c:xnnpack_plugin",
    ],
)

tflite_cc_library_with_c_headers_test(
    name = "stable_delegate",
    hdrs = ["stable_delegate.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tensorflow/lite/core/acceleration/configuration/c:stable_delegate",
    ],
)

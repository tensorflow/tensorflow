# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
load("//tensorflow/lite:build_def.bzl", "tflite_copts", "tflite_copts_warnings")
load(
    "//tensorflow/lite/core/shims:cc_library_with_tflite.bzl",
    "cc_library_with_tflite_with_c_headers_test",
)

# LINT.IfChange(tflite_acceleration_exported_headers)
exports_files([
    "delegate_plugin.h",
    "gpu_plugin.h",
    "xnnpack_plugin.h",
])
# LINT.ThenChange(
#     ../../../core/acceleration/configuration/c/BUILD:tflite_acceleration_exported_headers,
#     ../../../java/BUILD:tflite_acceleration_exported_headers
# )

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:private"],
    licenses = ["notice"],
)

cc_library_with_tflite_with_c_headers_test(
    name = "delegate_plugin",
    hdrs = ["delegate_plugin.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts() + tflite_copts_warnings(),
    generate_opaque_delegate_target = True,
    visibility = ["//visibility:public"],
    deps = ["//tensorflow/lite/core/acceleration/configuration/c:delegate_plugin"],
)

cc_library_with_tflite_with_c_headers_test(
    name = "nnapi_plugin",
    hdrs = ["nnapi_plugin.h"],
    copts = tflite_copts() + tflite_copts_warnings(),
    visibility = ["//visibility:public"],
    deps = ["//tensorflow/lite/core/acceleration/configuration/c:nnapi_plugin"],
)

test_suite(
    name = "nnapi_plugin_test",
    tests = [
        "//tensorflow/lite/core/acceleration/configuration/c:nnapi_plugin_test",
    ],
)

cc_library_with_tflite_with_c_headers_test(
    name = "gpu_plugin",
    hdrs = ["gpu_plugin.h"],
    copts = tflite_copts() + tflite_copts_warnings(),
    visibility = ["//visibility:public"],
    deps = ["//tensorflow/lite/core/acceleration/configuration/c:gpu_plugin"],
)

# For non-Android platforms, this should be built with '--copt=-DCL_DELEGATE_NO_GL'.
# On non-supported platforms (i.e. non-Android platforms if -DCL_DELEGATE_NO_GL wasn't specified),
# the test srcs are set to the empty list, so the test will succeed without testing anything.
test_suite(
    name = "gpu_plugin_test",
    tests = [
        "//tensorflow/lite/core/acceleration/configuration/c:gpu_plugin_test",
    ],
)

cc_library_with_tflite_with_c_headers_test(
    name = "xnnpack_plugin",
    hdrs = ["xnnpack_plugin.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts() + tflite_copts_warnings(),
    visibility = ["//visibility:public"],
    deps = ["//tensorflow/lite/core/acceleration/configuration/c:xnnpack_plugin"],
)

test_suite(
    name = "xnnpack_plugin_test",
    tests = [
        "//tensorflow/lite/core/acceleration/configuration/c:xnnpack_plugin_test",
    ],
)

cc_library_with_tflite_with_c_headers_test(
    name = "stable_delegate",
    hdrs = ["stable_delegate.h"],
    generate_opaque_delegate_target = True,
    visibility = ["//visibility:public"],
    deps = ["//tensorflow/lite/core/acceleration/configuration/c:stable_delegate"],
)

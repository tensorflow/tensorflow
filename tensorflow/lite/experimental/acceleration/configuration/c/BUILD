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

# LINT.IfChange

# C API for delegate plugins.

load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")

package(
    default_visibility = ["//visibility:private"],
    licenses = ["notice"],
)

cc_library(
    name = "delegate_plugin",
    hdrs = ["delegate_plugin.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/lite/core/experimental/acceleration/configuration/c:delegate_plugin",
    ],
)

cc_library(
    name = "nnapi_plugin",
    hdrs = ["nnapi_plugin.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/lite/core/experimental/acceleration/configuration/c:nnapi_plugin",
    ],
)

test_suite(
    name = "nnapi_plugin_test",
    tests = [
        "//tensorflow/lite/core/experimental/acceleration/configuration/c:nnapi_plugin_test",
    ],
)

cc_library(
    name = "gpu_plugin",
    hdrs = ["gpu_plugin.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/lite/core/experimental/acceleration/configuration/c:gpu_plugin",
    ],
)

# For non-Android platforms, this should be built with '--copt=-DCL_DELEGATE_NO_GL'.
# On non-supported platforms (i.e. non-Android platforms if -DCL_DELEGATE_NO_GL wasn't specified),
# the test srcs are set to the empty list, so the test will succeed without testing anything.
test_suite(
    name = "gpu_plugin_test",
    tests = [
        "//tensorflow/lite/core/experimental/acceleration/configuration/c:gpu_plugin_test",
    ],
)

cc_library(
    name = "xnnpack_plugin",
    hdrs = ["xnnpack_plugin.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/lite/core/experimental/acceleration/configuration/c:xnnpack_plugin",
    ],
)

test_suite(
    name = "xnnpack_plugin_test",
    tests = [
        "//tensorflow/lite/core/experimental/acceleration/configuration/c:xnnpack_plugin_test",
    ],
)

cc_library(
    name = "stable_delegate",
    hdrs = ["stable_delegate.h"],
    visibility = [
        "//tensorflow/lite:__subpackages__",
        "@org_tensorflow_lite_support//tensorflow_lite_support:__subpackages__",
    ],
    deps = [
        "//tensorflow/lite/core/experimental/acceleration/configuration/c:stable_delegate",
    ],
)

# LINT.ThenChange(BUILD)

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

load("//tensorflow:tensorflow.bzl", "get_compatible_with_portable")
load("//tensorflow/lite/delegates/gpu:build_defs.bzl", "gpu_delegate_linkopts")

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
        "//tensorflow/lite/c:common",
    ],
)

cc_library(
    name = "nnapi_plugin",
    srcs = ["nnapi_plugin.cc"],
    hdrs = ["nnapi_plugin.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":delegate_plugin",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/delegates/nnapi:nnapi_delegate",
        "//tensorflow/lite/experimental/acceleration/configuration:configuration_fbs",
        "//tensorflow/lite/experimental/acceleration/configuration:nnapi_plugin_impl",
    ],
)

cc_test(
    name = "nnapi_plugin_test",
    srcs = ["nnapi_plugin_test.cc"],
    deps = [
        ":nnapi_plugin",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/experimental/acceleration/configuration:configuration_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "gpu_plugin",
    srcs = ["gpu_plugin.cc"],
    hdrs = ["gpu_plugin.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":delegate_plugin",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/delegates/gpu:delegate",
        "//tensorflow/lite/experimental/acceleration/configuration:configuration_fbs",
        "//tensorflow/lite/experimental/acceleration/configuration:gpu_plugin_impl",
    ],
)

# For non-Android platforms, this should be built with '--copt=-DCL_DELEGATE_NO_GL'.
# On non-supported platforms (i.e. non-Android platforms if -DCL_DELEGATE_NO_GL wasn't specified),
# the test srcs are set to the empty list, so the test will succeed without testing anything.
cc_test(
    name = "gpu_plugin_test",
    srcs = select({
        "//tensorflow/lite/delegates/gpu:supports_gpu_delegate": ["gpu_plugin_test.cc"],
        "//conditions:default": [],
    }),
    linkopts = gpu_delegate_linkopts(),
    deps = select({
        "//tensorflow/lite/delegates/gpu:supports_gpu_delegate": [":gpu_plugin"],
        "//conditions:default": [],
    }) + [
        "@com_google_googletest//:gtest_main",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/experimental/acceleration/configuration:configuration_fbs",
    ],
)

cc_library(
    name = "xnnpack_plugin",
    srcs = ["xnnpack_plugin.cc"],
    hdrs = ["xnnpack_plugin.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":delegate_plugin",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/delegates/xnnpack:xnnpack_delegate",
        "//tensorflow/lite/experimental/acceleration/configuration:configuration_fbs",
    ],
)

cc_test(
    name = "xnnpack_plugin_test",
    srcs = ["xnnpack_plugin_test.cc"],
    deps = [
        ":xnnpack_plugin",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/delegates/xnnpack:xnnpack_delegate",
        "//tensorflow/lite/experimental/acceleration/configuration:configuration_fbs",
        "@com_google_googletest//:gtest_main",
        "@pthreadpool",
    ],
)

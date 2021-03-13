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

package(
    default_visibility = ["//visibility:private"],
    licenses = ["notice"],  # Apache 2.0
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

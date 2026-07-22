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

# Description:
# Java Native Interface (JNI) library intended for implementing the
# TensorFlow Lite Flex delegate for using TensorFlow ops with TensorFlow Lite.

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//tensorflow/lite:build_def.bzl", "tflite_copts")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:LICENSE"],
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

exports_files(
    srcs = ["flex_delegate_jni.cc"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "native",
    srcs = ["flex_delegate_jni.cc"],
    copts = tflite_copts(),
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        "//tensorflow/lite/delegates/flex:delegate",
        "//tensorflow/lite/delegates/utils:simple_delegate",
        "//tensorflow/lite/java/jni",
        "//tensorflow/lite/testing:init_tensorflow",
    ],
    alwayslink = 1,
)

# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

load("//tensorflow/lite:build_def.bzl", "tflite_copts", "tflite_linkopts")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(glob([
    "testdata/**",
]))

cc_library(
    name = "evaluation_stage",
    hdrs = ["evaluation_stage.h"],
    copts = tflite_copts(),
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/tools/evaluation/proto:evaluation_config_cc_proto",
    ],
)

cc_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    copts = tflite_copts(),
    deps = [
        "//tensorflow/lite:context",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/delegates/nnapi:nnapi_delegate",
    ] + select({
        "//tensorflow:android": [
            "//tensorflow/lite/delegates/gpu:delegate",
        ],
        "//conditions:default": [],
    }) + select({
        "//tensorflow:android_arm": [
            "//tensorflow/lite/experimental/delegates/hexagon:hexagon_delegate",
        ],
        "//tensorflow:android_arm64": [
            "//tensorflow/lite/experimental/delegates/hexagon:hexagon_delegate",
        ],
        "//conditions:default": [],
    }),
)

cc_test(
    name = "utils_test",
    srcs = ["utils_test.cc"],
    data = [
        "testdata/empty.txt",
        "testdata/labels.txt",
    ],
    linkopts = tflite_linkopts(),
    linkstatic = 1,
    deps = [
        ":utils",
        "//tensorflow/lite:context",
        "@com_google_googletest//:gtest_main",
    ],
)

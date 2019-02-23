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

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("//tensorflow/lite:build_def.bzl", "tflite_copts", "tflite_linkopts")

common_linkopts = tflite_linkopts() + select({
    "//conditions:default": [],
    "//tensorflow:android": [
        "-pie",
        "-llog",
    ],
})

cc_library(
    name = "evaluation_stage",
    srcs = ["evaluation_stage.cc"],
    hdrs = ["evaluation_stage.h"],
    copts = tflite_copts(),
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/lite/tools/evaluation/proto:evaluation_config_proto_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "evaluation_stage_factory",
    hdrs = ["evaluation_stage_factory.h"],
    copts = tflite_copts(),
    deps = [
        ":evaluation_stage",
        ":identity_stage",
        "//tensorflow/lite/tools/evaluation/proto:evaluation_config_proto_cc",
        "//tensorflow/lite/tools/evaluation/proto:evaluation_stages_proto_cc",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "identity_stage",
    srcs = ["identity_stage.cc"],
    hdrs = ["identity_stage.h"],
    copts = tflite_copts(),
    deps = [
        ":evaluation_stage",
        "@com_google_absl//absl/container:flat_hash_map",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:scope",
        "//tensorflow/lite/tools/evaluation/proto:evaluation_config_proto_cc",
    ] + select(
        {
            "//tensorflow:android": [
                "//tensorflow/core:android_tensorflow_lib",
            ],
            "//conditions:default": [
                "//tensorflow/core:tensorflow",
                "//tensorflow/core:protos_all_cc",
                "//tensorflow/core:core_cpu",
                "//tensorflow/core:framework",
                "//tensorflow/core:lib",
                "//tensorflow/core:ops",
            ],
        },
    ),
    alwayslink = 1,
)

tf_cc_test(
    name = "evaluation_stage_test",
    srcs = ["evaluation_stage_test.cc"],
    linkopts = common_linkopts,
    linkstatic = 1,
    tags = [
        "tflite_not_portable_android",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":evaluation_stage",
        ":evaluation_stage_factory",
        ":identity_stage",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/container:flat_hash_map",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/lite/tools/evaluation/proto:evaluation_config_proto_cc",
        "//tensorflow/lite/tools/evaluation/proto:evaluation_stages_proto_cc",
    ] + select(
        {
            "//tensorflow:android": [
                "//tensorflow/core:android_tensorflow_lib",
                "//tensorflow/core:android_tensorflow_test_lib",
            ],
            "//conditions:default": [
                "//tensorflow/core:framework",
                "//tensorflow/core:core_cpu",
                "//tensorflow/core:ops",
                "//tensorflow/core:tensorflow",
            ],
        },
    ),
)

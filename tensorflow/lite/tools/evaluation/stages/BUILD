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

load("//tensorflow/lite:build_def.bzl", "tflite_copts", "tflite_linkopts")

common_linkopts = tflite_linkopts() + select({
    "//conditions:default": [],
    "//tensorflow:android": [
        "-pie",
        "-llog",
    ],
})

exports_files(glob([
    "testdata/*.jpg",
]))

cc_library(
    name = "image_preprocessing_stage",
    srcs = ["image_preprocessing_stage.cc"],
    hdrs = ["image_preprocessing_stage.h"],
    copts = tflite_copts(),
    deps = [
        "@com_google_absl//absl/base",
        "//tensorflow/core:stats_calculator_portable",
        "//tensorflow/lite/profiling:time",
        "//tensorflow/lite/tools/evaluation:evaluation_stage",
        "//tensorflow/core:tflite_portable_logging",
        "//tensorflow/lite/kernels/internal:reference_base",
        "//tensorflow/lite/kernels/internal:types",
        "//tensorflow/lite/tools/evaluation/proto:evaluation_config_cc_proto",
        "//tensorflow/lite/tools/evaluation/proto:evaluation_stages_cc_proto",
    ] + select({
        "//tensorflow:android": [
            "//tensorflow/core:android_jpeg_internal",
        ],
        "//conditions:default": [
            "//tensorflow/core:jpeg_internal",
        ],
    }),
)

cc_test(
    name = "image_preprocessing_stage_test",
    srcs = ["image_preprocessing_stage_test.cc"],
    data = ["testdata/grace_hopper.jpg"],
    linkopts = common_linkopts,
    linkstatic = 1,
    deps = [
        ":image_preprocessing_stage",
        "//tensorflow/lite/tools/evaluation/proto:evaluation_config_cc_proto",
        "//tensorflow/lite/tools/evaluation/proto:evaluation_stages_cc_proto",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "topk_accuracy_eval_stage",
    srcs = ["topk_accuracy_eval_stage.cc"],
    hdrs = ["topk_accuracy_eval_stage.h"],
    copts = tflite_copts(),
    deps = [
        "//tensorflow/core:tflite_portable_logging",
        "//tensorflow/lite/tools/evaluation:evaluation_stage",
        "//tensorflow/lite/tools/evaluation/proto:evaluation_config_cc_proto",
        "//tensorflow/lite/tools/evaluation/proto:evaluation_stages_cc_proto",
    ],
)

cc_test(
    name = "topk_accuracy_eval_stage_test",
    srcs = ["topk_accuracy_eval_stage_test.cc"],
    linkopts = common_linkopts,
    linkstatic = 1,
    deps = [
        ":topk_accuracy_eval_stage",
        "//tensorflow/lite/tools/evaluation/proto:evaluation_config_cc_proto",
        "//tensorflow/lite/tools/evaluation/proto:evaluation_stages_cc_proto",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "tflite_inference_stage",
    srcs = ["tflite_inference_stage.cc"],
    hdrs = ["tflite_inference_stage.h"],
    copts = tflite_copts(),
    deps = [
        "//tensorflow/core:stats_calculator_portable",
        "//tensorflow/core:tflite_portable_logging",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/c:c_api_internal",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/profiling:time",
        "//tensorflow/lite/tools/evaluation:evaluation_stage",
        "//tensorflow/lite/tools/evaluation:utils",
        "//tensorflow/lite/tools/evaluation/proto:evaluation_config_cc_proto",
        "//tensorflow/lite/tools/evaluation/proto:evaluation_stages_cc_proto",
    ],
)

cc_test(
    name = "tflite_inference_stage_test",
    srcs = ["tflite_inference_stage_test.cc"],
    data = ["//tensorflow/lite:testdata/add_quantized.bin"],
    linkopts = common_linkopts,
    linkstatic = 1,
    deps = [
        ":tflite_inference_stage",
        "//tensorflow/lite/c:c_api_internal",
        "//tensorflow/lite/delegates/nnapi:nnapi_delegate",
        "//tensorflow/lite/tools/evaluation/proto:evaluation_config_cc_proto",
        "//tensorflow/lite/tools/evaluation/proto:evaluation_stages_cc_proto",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "image_classification_stage",
    srcs = ["image_classification_stage.cc"],
    hdrs = ["image_classification_stage.h"],
    copts = tflite_copts(),
    deps = [
        ":image_preprocessing_stage",
        ":tflite_inference_stage",
        ":topk_accuracy_eval_stage",
        "//tensorflow/core:tflite_portable_logging",
        "//tensorflow/lite/tools/evaluation:evaluation_stage",
        "//tensorflow/lite/tools/evaluation/proto:evaluation_config_cc_proto",
        "//tensorflow/lite/tools/evaluation/proto:evaluation_stages_cc_proto",
    ],
)

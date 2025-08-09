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

load("//tensorflow:strict.default.bzl", "py_strict_binary", "py_strict_library")
load("//tensorflow/core/platform:build_config_root.bzl", "if_pywrap")

# Building blocks for generating accuracy validation metric graphs.
load("//tensorflow/lite/experimental/acceleration/mini_benchmark:build_defs.bzl", "validation_model", "validation_test")
load("//tensorflow/lite/experimental/acceleration/mini_benchmark:special_rules.bzl", "minibenchmark_visibility_allowlist")

default_visibility_group = [
    "//tensorflow/lite/experimental/acceleration/mini_benchmark:__subpackages__",
    "//tensorflow/lite/tools/benchmark:__subpackages__",
] + minibenchmark_visibility_allowlist()

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = default_visibility_group,
    licenses = ["notice"],
)

py_strict_library(
    name = "kl_divergence",
    srcs = [
        "kl_divergence.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

py_strict_binary(
    name = "mobilenet",
    srcs = ["mobilenet.py"],
    deps = [
        ":kl_divergence",
        "//tensorflow:tensorflow_py",
        "//tensorflow/lite/python:lite",
        "//tensorflow/lite/tools:flatbuffer_utils",
    ] + if_pywrap(
        if_true = ["//tensorflow/python:_pywrap_tensorflow"],
    ),
)

genrule(
    name = "mobilenet_metrics_tflite",
    outs = ["mobilenet_metrics.tflite"],
    cmd = "$(location :mobilenet) $(location :mobilenet_metrics.tflite)",
    tools = [":mobilenet"],
)

validation_model(
    name = "mobilenet_quant_with_validation",
    testonly = 1,
    jpegs = "//tensorflow/lite/experimental/acceleration/mini_benchmark:odt_classifier_testfiles",
    main_model = "//tensorflow/lite/experimental/acceleration/mini_benchmark/models:mobilenet_v1_1.0_224_quant.tflite",
    metrics_model = ":mobilenet_metrics_tflite",
)

validation_test(
    name = "mobilenet_quant_validation_test",
    validation_model = ":mobilenet_quant_with_validation.tflite",
)

validation_model(
    name = "mobilenet_float_with_validation",
    testonly = 1,
    jpegs = "//tensorflow/lite/experimental/acceleration/mini_benchmark:odt_classifier_testfiles",
    main_model = "//tensorflow/lite/experimental/acceleration/mini_benchmark/models:mobilenet_v1_1.0_224.tflite",
    metrics_model = ":mobilenet_metrics_tflite",
)

validation_test(
    name = "mobilenet_float_validation_test",
    validation_model = ":mobilenet_float_with_validation.tflite",
)

py_strict_binary(
    name = "blazeface_metrics",
    srcs = ["blazeface_metrics.py"],
    deps = [
        ":kl_divergence",
        "//tensorflow:tensorflow_py",
        "//tensorflow/lite/python:lite",
        "//tensorflow/lite/tools:flatbuffer_utils",
    ] + if_pywrap(
        if_true = ["//tensorflow/python:_pywrap_tensorflow"],
    ),
)

genrule(
    name = "blazeface_metrics_tflite",
    outs = ["blazeface_metrics.tflite"],
    cmd = "$(location :blazeface_metrics) $(location :blazeface_metrics.tflite)",
    tools = [":blazeface_metrics"],
)

validation_model(
    name = "blazeface_mlkit_v1_with_validation",
    testonly = 1,
    jpegs = "//tensorflow/lite/experimental/acceleration/mini_benchmark:blazeface_testfiles",
    main_model = "//tensorflow/lite/experimental/acceleration/mini_benchmark/models:blazeface_mlkit_v1.tfl",
    metrics_model = ":blazeface_metrics_tflite",
    scale = "0.007812",
    zeropoint = "128",
)

validation_test(
    name = "blazeface_mlkit_v1_validation_regression_test",
    validation_model = ":blazeface_mlkit_v1_with_validation.tflite",
)

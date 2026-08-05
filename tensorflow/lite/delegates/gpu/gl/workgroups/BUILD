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

load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:LICENSE"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "calculator",
    srcs = ["calculator.cc"],
    hdrs = ["calculator.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:gpu_info",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/gl/compiler:shader_code",
    ],
)

cc_library(
    name = "default_calculator",
    srcs = ["default_calculator.cc"],
    hdrs = ["default_calculator.h"],
    deps = [
        ":calculator",
        "//tensorflow/lite/delegates/gpu/common:gpu_info",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/gl/compiler:shader_code",
    ],
)

cc_library(
    name = "calculator_from_metadata",
    srcs = ["calculator_from_metadata.cc"],
    hdrs = ["calculator_from_metadata.h"],
    deps = select({
        "//tensorflow/lite/delegates/gpu:tflite_gpu_binary_release": [],
        "//conditions:default": [
            ":calculator",
            ":default_calculator",
            "//tensorflow/lite/delegates/gpu/common:gpu_info",
            "//tensorflow/lite/delegates/gpu/common:types",
            "//tensorflow/lite/delegates/gpu/gl:common_cc_fbs",
            "//tensorflow/lite/delegates/gpu/gl:metadata_cc_fbs",
            "//tensorflow/lite/delegates/gpu/gl:workgroups_cc_fbs",
            "@com_google_absl//absl/container:flat_hash_map",
            "@com_google_absl//absl/memory",
            "@flatbuffers",
        ],
    }) + [
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/gl/compiler:shader_code",
    ],
)

cc_library(
    name = "best_effort_calculator",
    srcs = ["best_effort_calculator.cc"],
    hdrs = ["best_effort_calculator.h"],
    deps = [
        ":calculator",
        ":default_calculator",
        "//tensorflow/lite/delegates/gpu/common:gpu_info",
    ] + select({
        "//tensorflow/lite/delegates/gpu:tflite_gpu_binary_release": [],
        "//conditions:default": [
            ":calculator_from_metadata",
        ],
    }),
)

cc_library(
    name = "ideal_workgroup_picker",
    srcs = ["ideal_workgroup_picker.cc"],
    hdrs = ["ideal_workgroup_picker.h"],
    deps = [
        ":calculator",
        "//tensorflow/lite/delegates/gpu/common:gpu_info",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:types",
    ],
)

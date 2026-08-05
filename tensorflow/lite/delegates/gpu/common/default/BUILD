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
    name = "custom_parsers",
    srcs = ["custom_parsers.cc"],
    hdrs = ["//tensorflow/lite/delegates/gpu/common:custom_parsers.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:operation_parser",
        "//tensorflow/lite/delegates/gpu/common:unimplemented_operation_parser",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:any",
    ],
)

cc_library(
    name = "custom_transformations",
    srcs = ["custom_transformations.cc"],
    hdrs = ["//tensorflow/lite/delegates/gpu/common:custom_transformations.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:model_transformer",
    ],
)

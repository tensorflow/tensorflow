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
load("//tensorflow:tensorflow.bzl", "if_google")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

cc_library(
    name = "mla_test_utils",
    testonly = True,
    hdrs = ["mla_test_utils.h"],
    visibility = ["//visibility:private"],
    deps = [
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "@com_google_absl//absl/strings",
    ] + if_google([
        "//third_party/mira/mlarchive:mla",
        "//third_party/mira/mlarchive/google:google_env",
        "//tensorflow/cc/experimental/tfa:saved_model_converter",
        "//tensorflow/cc/experimental/tfa:test_utils",
    ]),
)

cc_library(
    name = "mla_utils",
    hdrs = ["mla_utils.h"],
    visibility = ["//visibility:private"],
    deps = [
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:statusor",
        "@com_google_absl//absl/strings",
    ] + if_google([
        "//third_party/mira/mlarchive:mla",
        "//third_party/mira/mlarchive:status_macro",
        "//tensorflow/cc/experimental/tfa:saved_model_converter",
        "//third_party/mira/mlarchive/google:google_env",
        "//tensorflow/core/platform:path",
        "//tensorflow/core/platform:status",
    ]),
)

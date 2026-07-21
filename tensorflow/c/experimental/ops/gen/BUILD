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
load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_binary",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "generate_cpp_main",
    srcs = ["generate_cpp_main.cc"],
    deps = [
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "//tensorflow/c/experimental/ops/gen/common",
        "//tensorflow/c/experimental/ops/gen/cpp/renderers",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/c/experimental/ops/gen/cpp",
        "//tensorflow/c/experimental/ops/gen/model",
        "//tensorflow/core:lib",
        # Without this line to link in ops, the global registry will be empty:
        "//tensorflow/core:ops",
    ],
)

tf_cc_binary(
    name = "generate_cpp",
    args = ["--stderrthreshold=1"],  # Warnings and errors to stderr.
    deps = [":generate_cpp_main"],
)

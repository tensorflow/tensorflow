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
# This package provides binaries that convert between multi-line and standard
# pbtxt (text-serialization of protocol message) files.

load("//tensorflow:tensorflow.bzl", "tf_cc_binary")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:private"],
    licenses = ["notice"],
)

exports_files(["placeholder.txt"])

tf_cc_binary(
    name = "tomlpbtxt",
    srcs = ["tomlpbtxt.cc"],
    deps = [
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:op_gen_lib",
        "@com_google_absl//absl/status",
    ],
)

tf_cc_binary(
    name = "frommlpbtxt",
    srcs = ["frommlpbtxt.cc"],
    deps = [
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:op_gen_lib",
        "@com_google_absl//absl/status",
    ],
)

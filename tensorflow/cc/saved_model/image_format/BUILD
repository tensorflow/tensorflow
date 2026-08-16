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

# On-disk serialization format for TensorFlow models.

load(
    "//tensorflow:tensorflow.bzl",
    "if_not_windows_or_mac",
)
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow/cc/experimental/tfa:__subpackages__",
        "//tensorflow/cc/saved_model:__subpackages__",
        "//tensorflow/tools/tfg_graph_transforms:__subpackages__",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "internal_api",
    srcs = ["internal_api.cc"],
    hdrs = ["internal_api.h"],
    deps = [
        "//tensorflow/cc/saved_model:metrics",
        "//tensorflow/cc/saved_model:util",
        "//tensorflow/core/platform",
        "//tensorflow/core/platform:env",
        "//tensorflow/core/protobuf:for_core_protos_cc",
        "//tensorflow/tools/proto_splitter/cc:max_size",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
    ] + if_not_windows_or_mac([
        "//tensorflow/tools/proto_splitter:merge",
        "//tensorflow/tools/proto_splitter/cc:saved_model_splitter",
    ]),
)

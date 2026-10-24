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

load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("//tensorflow:tensorflow.default.bzl", "filegroup")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

filegroup(
    name = "pywrap_required_hdrs",
    srcs = [
        "graph_verifier.h",
    ],
    visibility = [
        "//tensorflow/python/grappler:__pkg__",
    ],
)

cc_library(
    name = "graph_verifier",
    hdrs = [
        "graph_verifier.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "structure_verifier",
    srcs = ["structure_verifier.cc"],
    hdrs = [
        "structure_verifier.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":graph_verifier",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler/utils:topological_sort",
        "@com_google_absl//absl/status",
    ],
)

tf_cc_test(
    name = "structure_verifier_test",
    srcs = ["structure_verifier_test.cc"],
    deps = [
        ":structure_verifier",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//third_party/protobuf",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

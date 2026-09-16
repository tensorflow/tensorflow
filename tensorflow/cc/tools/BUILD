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

#include "third_party/absl/strings/str_cat.h"
#Description:
# TensorFlow cc tools.

load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
)
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "freeze_saved_model",
    srcs = ["freeze_saved_model.cc"],
    hdrs = ["freeze_saved_model.h"],
    deps = [
        "//tensorflow/cc/saved_model:loader",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
    ],
)

tf_cc_test(
    name = "freeze_saved_model_test",
    srcs = ["freeze_saved_model_test.cc"],
    deps = [
        ":freeze_saved_model",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:ops",
        "//tensorflow/cc:resource_variable_ops",
        "//tensorflow/cc:scope",
        "//tensorflow/cc/saved_model:loader",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:portable_gif_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensorflow",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/platform:status",
        "@com_google_absl//absl/status",
        "@xla//xla/tsl/platform:errors",
    ],
)

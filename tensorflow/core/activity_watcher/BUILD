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

load("//tensorflow:tensorflow.bzl", "if_not_mobile")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

filegroup(
    name = "activity_watcher_headers",
    srcs = [
        "activity.h",
        "activity_utils.h",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "activity_watcher",
    hdrs = ["activity.h"],
    defines = if_not_mobile(["TF_ENABLE_ACTIVITY_WATCHER"]),
    deps = [
        "//tensorflow/core:lib",
        "@com_google_absl//absl/container:flat_hash_map",
        "@xla//xla/tsl/platform:types",
    ] + if_not_mobile([
        ":activity_watcher_impl",
    ]),
    alwayslink = True,
)

cc_library(
    name = "activity_watcher_impl",
    srcs = [
        "activity.cc",
    ],
    hdrs = ["activity.h"],
    deps = [
        "//tensorflow/core:lib",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@xla//xla/tsl/platform:types",
    ],
    alwayslink = True,
)

cc_library(
    name = "activity_watcher_utils",
    srcs = ["activity_utils.cc"],
    hdrs = ["activity_utils.h"],
    deps = [
        ":activity_watcher",
        "//tensorflow/core:framework",
        "@com_google_absl//absl/strings",
        "@xla//xla/tsl/platform:types",
    ],
)

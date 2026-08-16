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
#   Libraries for storing tensors in SQL databases.

load("//tensorflow:tensorflow.bzl", "tf_cc_test", "tf_copts")
load(
    "//tensorflow/core/platform:rules_cc.bzl",
    "cc_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

cc_library(
    name = "sqlite",
    srcs = ["sqlite.cc"],
    hdrs = ["sqlite.h"],
    copts = tf_copts(),
    deps = [
        ":snapfn",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@org_sqlite",
    ],
)

cc_library(
    name = "snapfn",
    srcs = ["snapfn.cc"],
    copts = tf_copts() + ["-DSQLITE_OMIT_LOAD_EXTENSION"],
    linkstatic = 1,
    deps = [
        "@org_sqlite",
        "@snappy",
    ],
    alwayslink = 1,
)

tf_cc_test(
    name = "sqlite_test",
    size = "small",
    srcs = ["sqlite_test.cc"],
    deps = [
        ":sqlite",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

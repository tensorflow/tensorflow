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
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

cc_library(
    name = "checker",
    hdrs = ["checker.h"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core/profiler:protos_all_cc",
        "//tensorflow/core/profiler/internal:tfprof_stats",
    ],
)

cc_library(
    name = "internal_checker_runner_dummy",
    srcs = ["internal_checker_runner_dummy.cc"],
    hdrs = ["internal_checker_runner.h"],
    deps = [
        "//tensorflow/core/profiler:protos_all_cc",
        "//tensorflow/core/profiler/internal:tfprof_utils",
    ],
)

cc_library(
    name = "accelerator_utilization_checker",
    hdrs = ["accelerator_utilization_checker.h"],
    deps = [
        ":checker",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "operation_checker",
    hdrs = ["operation_checker.h"],
    deps = [
        ":checker",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "expensive_operation_checker",
    hdrs = ["expensive_operation_checker.h"],
    deps = [
        ":checker",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "tfprof_advisor",
    hdrs = ["tfprof_advisor.h"],
    deps = [
        ":accelerator_utilization_checker",
        ":checker",
        ":expensive_operation_checker",
        ":internal_checker_runner_dummy",
        ":operation_checker",
        "//tensorflow/core/profiler:protos_all_cc",
        "@com_google_absl//absl/strings:str_format",
    ],
)

tf_cc_test(
    name = "tfprof_advisor_test",
    srcs = ["tfprof_advisor_test.cc"],
    deps = [
        ":checker",
        ":tfprof_advisor",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core/framework:graph_proto_cc",
        "//tensorflow/core/framework:node_def_proto_cc",
        "//tensorflow/core/framework:step_stats_proto_cc",
        "//tensorflow/core/profiler:protos_all_cc",
        "//tensorflow/core/profiler/internal:tfprof_node",
        "//tensorflow/core/profiler/internal:tfprof_stats",
        "//tensorflow/core/profiler/internal:tfprof_tf_testlib",
        "@com_google_absl//absl/strings",
    ],
)

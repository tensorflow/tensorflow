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

load("@xla//xla/tsl:tsl.bzl", "internal_visibility")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//tensorflow/core/profiler/builds:build_config.bzl", "tf_profiler_copts")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow/core/profiler:internal"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//tensorflow/core/profiler:friends",
    ],
)

# DO NOT ADD NEW DEPENDENCIES TO ANY TARGET IN THIS FILE.
# Instead, use //third_party/xprof/utils.

cc_library(
    name = "math_utils",
    hdrs = ["math_utils.h"],
    visibility = internal_visibility([
        "//tensorflow/core/profiler/rpc:__pkg__",
    ]),
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@xla//xla/tsl/profiler/utils:math_utils",
    ],
)

cc_library(
    name = "trace_utils",
    hdrs = ["trace_utils.h"],
    copts = tf_profiler_copts(),
    visibility = internal_visibility([]),
    deps = [
        "@xla//xla/tsl/profiler/utils:trace_utils",
    ],
)

cc_library(
    name = "xplane_builder",
    hdrs = ["xplane_builder.h"],
    copts = tf_profiler_copts(),
    visibility = [":friends"],
    deps = [
        "@xla//xla/tsl/profiler/utils:xplane_builder",
    ],
)

cc_library(
    name = "xplane_schema",
    hdrs = ["xplane_schema.h"],
    copts = tf_profiler_copts(),
    visibility = [":friends"],
    deps = ["@xla//xla/tsl/profiler/utils:xplane_schema"],
)

cc_library(
    name = "xplane_utils",
    hdrs = ["xplane_utils.h"],
    copts = tf_profiler_copts(),
    visibility = [":friends"],
    deps = [
        "@xla//xla/tsl/profiler/utils:xplane_utils",
    ],
)

cc_library(
    name = "xplane_test_utils",
    testonly = True,
    hdrs = ["xplane_test_utils.h"],
    copts = tf_profiler_copts(),
    visibility = internal_visibility([]),
    deps = ["@xla//xla/tsl/profiler/utils:xplane_test_utils"],
)

cc_library(
    name = "xplane_visitor",
    hdrs = ["xplane_visitor.h"],
    copts = tf_profiler_copts(),
    visibility = [":friends"],
    deps = [
        "@xla//xla/tsl/profiler/utils:xplane_visitor",
    ],
)

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

load("@xla//xla/tsl:tsl.bzl", "if_google", "internal_visibility")
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
    name = "diagnostics",
    hdrs = if_google(["diagnostics.h"]),
    copts = tf_profiler_copts(),
    visibility = internal_visibility([]),
    deps = if_google([
        "@org_xprof//xprof/utils:diagnostics",
    ]),
)

cc_library(
    name = "event_span",
    hdrs = if_google(["event_span.h"]),
    copts = tf_profiler_copts(),
    visibility = internal_visibility([]),
    deps = if_google([
        "@org_xprof//xprof/utils:event_span",
    ]),
)

cc_library(
    name = "hardware_type_utils",
    hdrs = if_google(["hardware_type_utils.h"]),
    copts = tf_profiler_copts(),
    visibility = internal_visibility([]),
    deps = if_google([
        "@org_xprof//xprof/utils:hardware_type_utils",
    ]),
)

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
    name = "html_utils",
    hdrs = if_google(["html_utils.h"]),
    visibility = internal_visibility([]),
    deps = if_google([
        "@org_xprof//xprof/utils:html_utils",
    ]),
)

cc_library(
    name = "op_metrics_db_utils",
    hdrs = if_google(["op_metrics_db_utils.h"]),
    copts = tf_profiler_copts(),
    visibility = internal_visibility([]),
    deps = if_google([
        "@org_xprof//xprof/utils:op_metrics_db_utils",
    ]),
)

cc_library(
    name = "op_utils",
    hdrs = if_google(["op_utils.h"]),
    copts = tf_profiler_copts(),
    visibility = internal_visibility([]),
    deps = if_google([
        "@org_xprof//xprof/utils:op_utils",
    ]),
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

cc_library(
    name = "cost_utils",
    hdrs = if_google(["cost_utils.h"]),
    copts = tf_profiler_copts(),
    visibility = ["//visibility:private"],
    deps = if_google([
        "@org_xprof//xprof/utils:cost_utils",
    ]),
)

cc_library(
    name = "derived_timeline",
    hdrs = if_google(["derived_timeline.h"]),
    copts = tf_profiler_copts(),
    visibility = internal_visibility([]),
    deps = if_google([
        "@org_xprof//xprof/utils:derived_timeline",
    ]),
)

cc_library(
    name = "kernel_stats_utils",
    hdrs = if_google(["kernel_stats_utils.h"]),
    copts = tf_profiler_copts(),
    visibility = internal_visibility([]),
    deps = if_google(["@org_xprof//xprof/utils:kernel_stats_utils"]),
)

cc_library(
    name = "tfstreamz_utils",
    hdrs = if_google(["tfstreamz_utils.h"]),
    copts = tf_profiler_copts(),
    visibility = internal_visibility([]),
    deps = if_google([
        "@org_xprof//xprof/utils:tfstreamz_utils",
    ]),
)

cc_library(
    name = "step_intersection",
    hdrs = if_google(["step_intersection.h"]),
    copts = tf_profiler_copts(),
    visibility = internal_visibility([]),
    deps = if_google([
        "@org_xprof//xprof/utils:step_intersection",
    ]),
)

cc_library(
    name = "device_caps_utils",
    hdrs = if_google(["device_caps_utils.h"]),
    copts = tf_profiler_copts(),
    visibility = internal_visibility([]),
    deps = if_google(["@org_xprof//xprof/utils:device_caps_utils"]),
)

cc_library(
    name = "gpu_event_stats",
    hdrs = if_google(["gpu_event_stats.h"]),
    copts = tf_profiler_copts(),
    visibility = internal_visibility([]),
    deps = if_google([
        "@org_xprof//xprof/utils:gpu_event_stats",
    ]),
)

cc_library(
    name = "hlo_proto_map",
    hdrs = if_google(["hlo_proto_map.h"]),
    visibility = internal_visibility([]),
    deps = if_google([
        "@org_xprof//xprof/utils:hlo_proto_map",
    ]),
)

cc_library(
    name = "hlo_proto_to_module",
    hdrs = if_google(["hlo_proto_to_module.h"]),
    visibility = internal_visibility([]),
    deps = if_google([
        "@org_xprof//xprof/utils:hlo_proto_to_module",
    ]),
)

cc_library(
    name = "hlo_module_map",
    hdrs = if_google(["hlo_module_map.h"]),
    visibility = internal_visibility([]),
    deps = if_google([
        "@org_xprof//xprof/utils:hlo_module_map",
    ]),
)

cc_library(
    name = "hlo_module_utils",
    hdrs = if_google(["hlo_module_utils.h"]),
    visibility = internal_visibility([
        "//third_party/odml/model_explorer/backend/adapters/hlo:__pkg__",
    ]),
    deps = if_google([
        "@org_xprof//xprof/utils:hlo_module_utils",
    ]),
)

cc_library(
    name = "xprof_gpu_cost_analysis",
    hdrs = if_google(["xprof_gpu_cost_analysis.h"]),
    visibility = ["//visibility:private"],
    deps = if_google([
        "@org_xprof//xprof/utils:xprof_gpu_cost_analysis",
    ]),
)

cc_library(
    name = "tpu_step_breakdown_utils",
    hdrs = if_google(["tpu_step_breakdown_utils.h"]),
    visibility = internal_visibility([]),
    deps = if_google([
        "@org_xprof//xprof/utils:tpu_step_breakdown_utils",
    ]),
)

cc_library(
    name = "tpu_step_details_utils",
    hdrs = if_google(["tpu_step_details_utils.h"]),
    visibility = internal_visibility([]),
    deps = if_google([
        "@org_xprof//xprof/utils:tpu_step_details_utils",
    ]),
)

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

load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//tensorflow/core/profiler/builds:build_config.bzl", "tf_profiler_copts")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow/core/profiler:internal"],
    licenses = ["notice"],
)

cc_library(
    name = "xplane_to_step_stats",
    srcs = ["xplane_to_step_stats.cc"],
    hdrs = ["xplane_to_step_stats.h"],
    copts = tf_profiler_copts(),
    deps = [
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/profiler/utils:gpu_event_stats",
        "//tensorflow/core/profiler/utils:xplane_schema",
        "//tensorflow/core/profiler/utils:xplane_utils",
        "//tensorflow/core/profiler/utils:xplane_visitor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@org_xprof//xprof/utils:gpu_event_stats",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@xla//xla/tsl/profiler/utils:math_utils",
        "@xla//xla/tsl/profiler/utils:tf_xplane_visitor",
    ],
)

# DO NOT ADD NEW DEPENDENCIES TO ANY TARGET IN THIS FILE.
# Instead, use //third_party/xprof/convert.

cc_library(
    name = "hlo_proto_to_memory_visualization_utils",
    hdrs = ["hlo_proto_to_memory_visualization_utils.h"],
    copts = tf_profiler_copts(),
    visibility = [
        "//learning/deepmind/jax/statix:__subpackages__",
        "//smartass/brain/tpu_worker:__subpackages__",
        "//tensorflow/core/profiler:internal",
    ],
    deps = [
        "@org_xprof//xprof/convert:hlo_proto_to_memory_visualization_utils",
    ],
)

cc_library(
    name = "profile_time_breakdown",
    hdrs = ["profile_time_breakdown.h"],
    copts = tf_profiler_copts(),
    visibility = [
        "//platforms/performance/autograppler/utils:__subpackages__",
        "//tensorflow/core/profiler:internal",
    ],
    deps = [
        "@org_xprof//xprof/convert:profile_time_breakdown",
    ],
)

cc_library(
    name = "xplane_to_op_stats",
    hdrs = ["xplane_to_op_stats.h"],
    copts = tf_profiler_copts(),
    visibility = [
        "//platforms/xla/tools/multihost_hlo_runner/hybrid_sim:__subpackages__",
        "//tensorflow/core/profiler:internal",
    ],
    deps = [
        "@org_xprof//xprof/convert:xplane_to_op_stats",
    ],
)

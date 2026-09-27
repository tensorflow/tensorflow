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

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

cc_library(
    name = "trace_events_filter_interface",
    hdrs = ["trace_events_filter_interface.h"],
    deps = [
        "@org_xprof//xprof/convert/trace_viewer:trace_events_filter_interface",
    ],
)

cc_library(
    name = "trace_viewer_visibility",
    hdrs = ["trace_viewer_visibility.h"],
    deps = [
        "@org_xprof//xprof/convert/trace_viewer:trace_viewer_visibility",
    ],
)

cc_library(
    name = "trace_viewer_color",
    hdrs = ["trace_viewer_color.h"],
    deps = [
        "@org_xprof//xprof/convert/trace_viewer:trace_viewer_color",
    ],
)

cc_library(
    name = "trace_events_to_json",
    hdrs = ["trace_events_to_json.h"],
    deps = [
        "@org_xprof//xprof/convert/trace_viewer:trace_events_to_json",
    ],
)

cc_library(
    name = "trace_event_arguments_builder",
    hdrs = ["trace_event_arguments_builder.h"],
    deps = [
        "@org_xprof//xprof/convert/trace_viewer:trace_event_arguments_builder",
    ],
)

cc_library(
    name = "trace_events_util",
    hdrs = ["trace_events_util.h"],
    deps = [
        "@org_xprof//xprof/convert/trace_viewer:trace_events_util",
    ],
)

cc_library(
    name = "trace_events",
    hdrs = ["trace_events.h"],
    deps = [
        "@org_xprof//xprof/convert/trace_viewer:trace_events",
    ],
)

# Copyright 2026 The OpenXLA Authors. All Rights Reserved.
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

load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")
load("//xla/tsl/profiler/builds:build_config.bzl", "tf_profiler_copts")

# copybara:uncomment package(default_applicable_licenses = ["//tensorflow:license"])

cc_library(
    name = "ondevice_trace_event",
    hdrs = ["ondevice_trace_event.h"],
    copts = tf_profiler_copts(),
    visibility = internal_visibility([
        "//xla/python:__pkg__",
        "//xla/tsl/profiler:internal",
        "//xla/tsl/profiler:xla_profiler_backends",
        "//tensorflow/lite:__pkg__",
    ]),
    deps = [
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "ondevice_event_collector",
    hdrs = ["ondevice_event_collector.h"],
    copts = tf_profiler_copts(),
    visibility = internal_visibility([
        "//xla/python:__pkg__",
        "//xla/tsl/profiler:internal",
        "//xla/tsl/profiler:xla_profiler_backends",
        "//tensorflow/lite:__pkg__",
    ]),
    deps = [
        ":ondevice_trace_event",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "ondevice_event_receiver",
    srcs = ["ondevice_event_receiver.cc"],
    hdrs = ["ondevice_event_receiver.h"],
    copts = tf_profiler_copts(),
    visibility = internal_visibility([
        "//xla/python:__pkg__",
        "//xla/tsl/profiler:internal",
        "//xla/tsl/profiler:xla_profiler_backends",
        "//tensorflow/lite:__pkg__",
    ]),
    deps = [
        ":ondevice_event_collector",
        ":ondevice_trace_event",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
    ],
)

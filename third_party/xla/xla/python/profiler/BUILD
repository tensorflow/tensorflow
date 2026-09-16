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

load("//xla/tsl:tsl.default.bzl", "tsl_pybind_extension")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

# copybara:uncomment package(default_applicable_licenses = ["//tensorflow:license"])

cc_library(
    name = "profile_data_lib",
    srcs = ["profile_data_lib.cc"],
    hdrs = ["profile_data_lib.h"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    visibility = [
        "//xla/python:__pkg__",
    ],
    deps = [
        "//xla/pjrt:status_casters",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:logging",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_protobuf//:protobuf_lite",
        "@nanobind",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:protobuf",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
    ],
    alwayslink = 1,
)

tsl_pybind_extension(
    name = "profile_data",
    srcs = ["py_profile_data.cc"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    deps = [
        ":profile_data_lib",
        "@com_google_protobuf//:protobuf",
        "@nanobind",
        "@tsl//tsl/platform:protobuf",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
    ],
)

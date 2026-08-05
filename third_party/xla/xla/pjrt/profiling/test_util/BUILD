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

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load(
    "//xla/tsl:tsl.bzl",
    "internal_visibility",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([
        "//xla:internal",
    ]),
    licenses = ["notice"],
)

cc_library(
    name = "mock_device_time_measurement",
    testonly = True,
    srcs = [
        "mock_device_time_measurement.cc",
        "//xla/pjrt/profiling:device_time_measurement.h",
    ],
    hdrs = ["mock_device_time_measurement.h"],
    # copybara:uncomment_begin(google-only)
    # compatible_with = ["//buildenv/target:non_prod"],
    # copybara:uncomment_end
    deps = [
        "//xla/pjrt:pjrt_compiler",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/debugging:leak_check",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

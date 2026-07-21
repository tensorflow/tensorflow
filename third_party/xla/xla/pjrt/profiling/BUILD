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

# Device runtime profiling.
load(
    "//xla/tsl:tsl.bzl",
    "internal_visibility",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([
        "//learning/brain/google/runtime:__subpackages__",
        "//learning/brain/tpu/runtime/hal/internal:__subpackages__",
        "//learning/serving/model_servers:__subpackages__",
        "//xla:internal",
    ]),
    licenses = ["notice"],
)

exports_files(
    [
        "device_time_measurement.h",
        "no_op_device_time_measurement.h",
    ],
    visibility = internal_visibility([
        "//learning/brain/google/runtime:__subpackages__",
        "//learning/brain/tpu/runtime/hal/internal:__subpackages__",
        "//tensorflow:__subpackages__",
    ]),
)

cc_library(
    name = "no_op_device_time_measurement",
    srcs = [
        "device_time_measurement.h",
        "no_op_device_time_measurement.cc",
        "no_op_device_time_measurement.h",
    ],
    # copybara:uncomment_begin(google-only)
    # compatible_with = ["//buildenv/target:non_prod"],
    # copybara:uncomment_end
    deps = [
        "//xla/pjrt:pjrt_compiler",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "device_time_measurement",
    # copybara:uncomment_begin(google-only)
    # compatible_with = ["//buildenv/target:non_prod"],
    # copybara:uncomment_end
    textual_hdrs = ["device_time_measurement.h"],
    deps = [
        # copybara:comment_begin(oss-only)
        ":no_op_device_time_measurement",
        # copybara:comment_end
        # copybara:uncomment_begin(google-only)
        # "//learning/brain/google/runtime:device_runtime_profiling",  # buildcleaner: keep
        # copybara:uncomment_end
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "//xla/pjrt:pjrt_compiler",
    ],
)

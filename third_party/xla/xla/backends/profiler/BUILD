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
    "if_with_tpu_support",
    "internal_visibility",
)

# copybara:uncomment package(default_applicable_licenses = ["//tensorflow:license"])

package_group(
    name = "friends",
    packages = [
        "//xla/backends/profiler/cpu",
        "//xla/backends/profiler/gpu",
        "//xla/backends/profiler/tpu",
    ],
)

cc_library(
    name = "profiler_backends",
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    visibility = internal_visibility(["//xla:internal"]),
    deps = [
        "//xla/backends/profiler/cpu:host_tracer",
        "//xla/backends/profiler/cpu:metadata_collector",
    ] + if_with_tpu_support(
        [
            "//xla/backends/profiler/tpu:tpu_tracer",
        ],
    ),
    alwayslink = True,
)

cc_library(
    name = "simple_multipass_tracer",
    srcs = ["simple_multipass_tracer.cc"],
    hdrs = ["simple_multipass_tracer.h"],
    visibility = [
        "//xla:internal",
        "//xla/tsl/profiler:friends",
    ],
    deps = [
        "//xla/tsl/platform:logging",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/profiler/lib:profiler_collection",
        "@tsl//tsl/profiler/lib:profiler_factory",
        "@tsl//tsl/profiler/lib:profiler_interface",
        "@tsl//tsl/profiler/protobuf:profiler_options_proto_cc",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
    ],
    alwayslink = True,
)

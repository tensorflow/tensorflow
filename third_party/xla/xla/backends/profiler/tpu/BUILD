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

load("//xla/tsl:tsl.bzl", "if_with_tpu_support")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")
load("//xla/tsl/profiler/builds:build_config.bzl", "tf_profiler_copts")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

cc_library(
    name = "tpu_tracer",
    srcs = if_with_tpu_support(["tpu_tracer.cc"]),
    copts = tf_profiler_copts(),
    visibility = [
        "//xla:internal",
        "//xla/tsl/profiler:friends",
    ],
    deps = [
        "//xla/tpu:tpu_api",
        "//xla/tpu:tpu_api_dlsym_set_fn",
        "//xla/tpu:tpu_ops_c_api_hdrs",
        "//xla/tpu:tpu_profiler_init_fns",
        "//xla/tpu:tsl_status_helper",
        "//xla/tsl/c:tsl_status",
        "//xla/tsl/profiler/utils:xplane_schema",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:status",
        "@tsl//tsl/platform:types",
        "@tsl//tsl/profiler/lib:profiler_factory",
        "@tsl//tsl/profiler/lib:profiler_interface",
        "@tsl//tsl/profiler/protobuf:profiler_options_proto_cc",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
    ],
    alwayslink = True,
)

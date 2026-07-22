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

load(
    "//xla:xla.default.bzl",
    "xla_cc_binary",
    "xla_cc_test",
)
load(
    "//xla/tsl/platform:build_config.bzl",
    "tf_proto_library",
)
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:private"],
    licenses = ["notice"],
)

tf_proto_library(
    name = "split_proto_cli_proto",
    srcs = ["split_proto_cli.proto"],
    deps = [
        "//xla/pjrt/proto:compile_options_proto",
        "//xla/service/gpu:gpu_executable_proto",
    ],
)

cc_library(
    name = "split_proto_cli_lib",
    srcs = ["split_proto_cli_lib.cc"],
    hdrs = ["split_proto_cli_lib.h"],
    compatible_with = [],
    deps = [
        ":split_proto_cli_proto_cc",
        "//xla:xla_data_proto_cc",
        "//xla/backends/gpu/codegen/kernels:custom_kernel_proto_cc",
        "//xla/backends/gpu/runtime:thunk_proto_cc",
        "//xla/pjrt/proto:compile_options_proto_cc",
        "//xla/service:hlo_proto_cc",
        "//xla/service/cpu:executable_proto_cc",
        "//xla/service/gpu:gpu_executable_proto_cc",
        "//xla/stream_executor:device_description_proto_cc",
        "//xla/stream_executor/abi:executable_abi_version_proto_cc",
        "//xla/tsl/platform:status_macros",
        "//xla/tsl/util:fixed_option_set_flag",
        "//xla/tsl/util:sorted_range",
        "//xla/util/split_proto:split_executable_and_options_writer",
        "//xla/util/split_proto:split_gpu_executable_writer",
        "//xla/util/split_proto:split_hlo_writer",
        "//xla/util/split_proto:split_proto_cc",
        "//xla/util/split_proto:split_proto_reader",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_protobuf//:protobuf",
        "@com_google_protobuf//:protobuf_lite",
        "@riegeli//riegeli/base:initializer",
        "@riegeli//riegeli/base:types",
        "@riegeli//riegeli/bytes:reader",
        "@riegeli//riegeli/bytes:string_reader",
        "@riegeli//riegeli/bytes:string_writer",
        "@riegeli//riegeli/bytes:writer",
        "@riegeli//riegeli/messages:parse_message",
        "@riegeli//riegeli/messages:serialize_message",
        "@riegeli//riegeli/records:record_reader",
    ],
)

xla_cc_binary(
    name = "split_proto_cli",
    srcs = ["split_proto_cli.cc"],
    deps = [
        ":split_proto_cli_lib",
        "//xla/service:riegeli_file_reader_factory",
        "//xla/service:riegeli_file_writer_factory",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:status_macros",
        "//xla/tsl/util:command_line_flags",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@riegeli//riegeli/base:initializer",
        "@riegeli//riegeli/bytes:reader",
        "@riegeli//riegeli/bytes:std_io",
        "@riegeli//riegeli/bytes:writer",
        "@tsl//tsl/platform:platform_port",
    ],
)

xla_cc_test(
    name = "split_proto_cli_test",
    srcs = ["split_proto_cli_test.cc"],
    deps = [
        ":split_proto_cli_lib",
        ":split_proto_cli_proto_cc",
        "//xla:xla_data_proto_cc",
        "//xla/backends/gpu/runtime:thunk_proto_cc",
        "//xla/pjrt/proto:compile_options_proto_cc",
        "//xla/service:custom_call_target_registry",
        "//xla/service:hlo_proto_cc",
        "//xla/service/gpu:dense_data_intermediate_proto_cc",
        "//xla/service/gpu:gpu_executable_proto_cc",
        "//xla/stream_executor:device_description_proto_cc",
        "//xla/stream_executor/cuda:cuda_compute_capability_proto_cc",
        "//xla/tsl/util/proto:parse_text_proto",
        "//xla/tsl/util/proto:proto_matchers",
        "//xla/util/split_proto:split_executable_and_options_writer",
        "//xla/util/split_proto:split_gpu_executable_writer",
        "//xla/util/split_proto:split_proto_reader",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest_main",
        "@com_google_protobuf//:protobuf",
        "@riegeli//riegeli/base:initializer",
        "@riegeli//riegeli/bytes:string_reader",
        "@riegeli//riegeli/bytes:string_writer",
    ],
)

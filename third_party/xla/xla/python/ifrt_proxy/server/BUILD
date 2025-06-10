# Copyright 2023 The OpenXLA Authors.
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
load("//xla/python/ifrt_proxy/common:ifrt_proxy.bzl", "cc_library", "default_ifrt_proxy_visibility", "ifrt_proxy_cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = default_ifrt_proxy_visibility,
)

cc_library(
    name = "grpc_server",
    srcs = ["grpc_server.cc"],
    hdrs = ["grpc_server.h"],
    deps = [
        ":grpc_service_impl",
        ":host_buffer",
        ":ifrt_backend",
        "//xla/python/ifrt",
        "//xla/python/ifrt:attribute_map",
        "//xla/python/ifrt_proxy/common:grpc_credentials_possibly_insecure_wrapper",
        "//xla/python/ifrt_proxy/common:grpc_ifrt_service_cc_grpc_proto",
        "//xla/python/ifrt_proxy/common:ifrt_service_proto_cc",
        "//xla/tsl/platform:statusor",
        "@com_github_grpc_grpc//:grpc",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
    ],
)

ifrt_proxy_cc_test(
    name = "grpc_server_test",
    srcs = ["grpc_server_test.cc"],
    tags = ["not_run:arm"],  # TODO(b/326080238): Fix this.
    deps = [
        ":grpc_server",
        "//xla/python/ifrt_proxy/common:grpc_ifrt_service_cc_grpc_proto",
        "//xla/tsl/platform:status_matchers",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "grpc_service_impl",
    srcs = ["grpc_service_impl.cc"],
    hdrs = ["grpc_service_impl.h"],
    deps = [
        ":host_buffer",
        ":ifrt_backend",
        ":version",
        "//xla/pjrt/distributed:util",
        "//xla/python/ifrt:attribute_map",
        "//xla/python/ifrt_proxy/common:grpc_ifrt_service_cc_grpc_proto",
        "//xla/python/ifrt_proxy/common:grpc_ifrt_service_proto_cc",
        "//xla/python/ifrt_proxy/common:ifrt_service_proto_cc",
        "//xla/python/ifrt_proxy/common:proto_util",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/log:die_if_null",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@local_tsl//tsl/profiler/lib:traceme",
    ],
)

ifrt_proxy_cc_test(
    name = "grpc_service_impl_test",
    size = "small",
    srcs = ["grpc_service_impl_test.cc"],
    tags = ["not_run:arm"],  # TODO(b/326080238): Fix this.
    deps = [
        ":grpc_server",
        ":grpc_service_impl",
        ":host_buffer",
        ":version",
        "//xla/python/ifrt:attribute_map",
        "//xla/python/ifrt_proxy/client:grpc_host_buffer",
        "//xla/python/ifrt_proxy/common:grpc_ifrt_service_cc_grpc_proto",
        "//xla/python/ifrt_proxy/common:ifrt_service_proto_cc",
        "//xla/tsl/platform:status_matchers",
        "//xla/tsl/platform:test",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "ifrt_backend",
    srcs = ["ifrt_backend.cc"],
    hdrs = ["ifrt_backend.h"],
    deps = [
        ":host_buffer",
        ":host_callback",
        ":version",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:xla_data_proto_cc",
        "//xla/pjrt:pjrt_layout",
        "//xla/python/ifrt",
        "//xla/python/ifrt:basic_device_list",
        "//xla/python/ifrt:program_serdes",
        "//xla/python/ifrt:serdes",
        "//xla/python/ifrt:sharding_serdes",
        "//xla/python/ifrt_proxy/common:array_util",
        "//xla/python/ifrt_proxy/common:common_serdes",
        "//xla/python/ifrt_proxy/common:ifrt_service_proto_cc",
        "//xla/python/ifrt_proxy/common:prof_util",
        "//xla/python/ifrt_proxy/common:proto_util",
        "//xla/python/ifrt_proxy/common:types",
        "//xla/python/ifrt_proxy/common:types_proto_cc",
        "//xla/python/ifrt_proxy/common:versions",
        "//xla/python/pjrt_ifrt:xla_ifrt",
        "//xla/tsl/concurrency:ref_count",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:status_to_from_proto",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/functional:bind_front",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
    ],
)

ifrt_proxy_cc_test(
    name = "ifrt_backend_test",
    srcs = ["ifrt_backend_test.cc"],
    deps = [
        ":host_buffer",
        ":host_callback",
        ":ifrt_backend",
        ":version",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/testlib:test",
        "//xla/pjrt:host_callback",
        "//xla/pjrt:pjrt_layout",
        "//xla/python/ifrt",
        "//xla/python/ifrt:attribute_map",
        "//xla/python/ifrt:basic_device_list",
        "//xla/python/ifrt:mock",
        "//xla/python/ifrt:program_serdes",
        "//xla/python/ifrt:serdes",
        "//xla/python/ifrt:sharding_serdes",
        "//xla/python/ifrt_proxy/common:array_util",
        "//xla/python/ifrt_proxy/common:ifrt_service_proto_cc",
        "//xla/python/ifrt_proxy/common:types_proto_cc",
        "//xla/python/ifrt_proxy/common:versions",
        "//xla/python/pjrt_ifrt:xla_ifrt",
        "//xla/service:computation_placer_hdr",
        "//xla/tsl/concurrency:ref_count",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:status_matchers",
        "//xla/tsl/platform:status_to_from_proto",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "//xla/tsl/protobuf:error_codes_proto_impl_cc",
        "//xla/tsl/protobuf:status_proto_cc",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@local_tsl//tsl/platform:protobuf",
    ],
)

cc_library(
    name = "mock_ifrt_backend",
    testonly = True,
    hdrs = ["mock_ifrt_backend.h"],
    deps = [
        ":ifrt_backend",
        "//xla/python/ifrt",
        "//xla/python/ifrt_proxy/common:ifrt_service_proto_cc",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_for_library",
    ],
)

cc_library(
    name = "host_buffer",
    srcs = ["host_buffer.cc"],
    hdrs = ["host_buffer.h"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@local_tsl//tsl/profiler/lib:traceme",
    ],
)

cc_library(
    name = "host_callback",
    srcs = ["host_callback.cc"],
    hdrs = ["host_callback.h"],
    deps = [
        "//xla:shape_util",
        "//xla/pjrt:host_callback",
        "//xla/python/ifrt",
        "//xla/python/ifrt_proxy/common:proto_util",
        "//xla/python/pjrt_ifrt",
        "//xla/python/pjrt_ifrt:xla_host_callback_proto_cc",
        "//xla/tsl/concurrency:ref_count",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/functional:bind_front",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "version",
    srcs = ["version.cc"],
    hdrs = ["version.h"],
    deps = [
        "//xla/python/ifrt_proxy/common:versions",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

ifrt_proxy_cc_test(
    name = "version_test",
    srcs = ["version_test.cc"],
    deps = [
        ":version",
        "//xla/tsl/platform:status_matchers",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)

ifrt_proxy_cc_test(
    name = "host_buffer_test",
    srcs = ["host_buffer_test.cc"],
    deps = [
        ":host_buffer",
        "//xla/python/ifrt",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:status_matchers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
    ],
)

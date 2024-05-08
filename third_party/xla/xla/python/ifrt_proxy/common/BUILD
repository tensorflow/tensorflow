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
load("//xla/tsl:tsl.bzl", "if_google")
load("//xla/tsl/platform:build_config.bzl", "tf_proto_library")
# copybara:uncomment load("@bazel_skylib//:bzl_library.bzl", "bzl_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = default_ifrt_proxy_visibility,
)

# Export headers referenced by the google-internal-version of grpc_credentials.
exports_files(
    ["grpc_credentials.h"],
    visibility = if_google(
        ["//xla/python/ifrt_proxy/common/google:__pkg__"],
        ["//visibility:private"],
    ),
)

cc_library(
    name = "grpc_credentials",
    hdrs = ["grpc_credentials.h"],
    deps = if_google(
        ["//xla/python/ifrt_proxy/common/google:grpc_credentials_lib"],
        [":grpc_credentials_oss_lib"],
    ) + ["@com_github_grpc_grpc//:grpc++"],
)

cc_library(
    name = "grpc_credentials_oss_lib",
    srcs = [
        "grpc_credentials.cc",
        "grpc_credentials.h",
    ],
    visibility = ["//visibility:private"],
    deps = [
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@local_tsl//tsl/platform",
    ],
    alwayslink = True,
)

cc_library(
    name = "grpc_credentials_possibly_insecure_wrapper",
    srcs = [
        "grpc_credentials_possibly_insecure_wrapper.cc",
    ],
    hdrs = [
        "grpc_credentials_possibly_insecure_wrapper.h",
    ],
    visibility = ["//xla/python/ifrt_proxy:__subpackages__"],
    deps = [
        ":grpc_credentials",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/log",
        "@local_tsl//tsl/platform",
    ],
)

tf_proto_library(
    name = "types_proto",
    srcs = ["types.proto"],
)

tf_proto_library(
    name = "ifrt_service_proto",
    srcs = ["ifrt_service.proto"],
    protodeps = [
        ":types_proto",
        # copybara:uncomment "@com_google_protobuf//:any",
        "//xla:xla_data_proto",
        "//xla/pjrt/proto:execute_options_proto",
        "//xla/python/ifrt:array_spec_proto",
        "//xla/python/ifrt:attribute_map_proto",
        "//xla/python/ifrt:dtype_proto",
        "//xla/python/ifrt:execute_options_proto",
        "//xla/python/ifrt:remap_plan_proto",
        "//xla/python/ifrt:serdes_proto",
        "//xla/python/ifrt:shape_proto",
        "//xla/python/ifrt:sharding_proto",
        "//xla/tsl/protobuf:status_proto",
    ],
)

tf_proto_library(
    name = "grpc_ifrt_service_proto",
    srcs = ["grpc_ifrt_service.proto"],
    has_services = True,
    create_go_proto = False,
    create_grpc_library = True,
    create_java_proto = False,
    create_kotlin_proto = False,
    protodeps = [
        ":ifrt_service_proto",
        "//xla/python/ifrt:attribute_map_proto",
    ],
)

cc_library(
    name = "types",
    srcs = ["types.cc"],
    hdrs = ["types.h"],
    deps = [
        ":ifrt_service_proto_cc",
        ":types_proto_cc",
        "//xla/pjrt:pjrt_common",
        "//xla/python/ifrt",
        "//xla/python/ifrt:serdes",
        "//xla/python/ifrt:sharding_serdes",
        "//xla/tsl/concurrency:ref_count",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
    ] + if_google(["@com_google_absl//absl/types:source_location"]),
)

ifrt_proxy_cc_test(
    name = "types_test",
    srcs = ["types_test.cc"],
    deps = [
        ":types",
        ":types_proto_cc",
        "//xla/pjrt:pjrt_common",
        "//xla/tsl/platform:status_matchers",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_proto_library(
    name = "array_util_proto",
    srcs = ["array_util.proto"],
)

cc_library(
    name = "array_util",
    srcs = ["array_util.cc"],
    hdrs = ["array_util.h"],
    deps = [
        ":array_util_proto_cc",
        "//xla/python/ifrt",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/types:span",
    ],
)

ifrt_proxy_cc_test(
    name = "array_util_test",
    srcs = ["array_util_test.cc"],
    deps = [
        ":array_util",
        "//xla/python/ifrt",
        "//xla/tsl/platform:status_matchers",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

# common_serdes is a collection of all common libraries that register SerDes implementations.
cc_library(
    name = "common_serdes",
    deps = [
        "//xla/python/ifrt:custom_call_program_serdes",
        "//xla/python/ifrt:plugin_program_serdes",
        "//xla/python/ifrt/hlo:hlo_program_serdes",
        "//xla/python/ifrt/ir:ifrt_ir_program_serdes",
        "//xla/python/pjrt_ifrt:xla_sharding_serdes",
    ],
    alwayslink = True,
)

cc_library(
    name = "proto_util",
    srcs = ["proto_util.cc"],
    hdrs = ["proto_util.h"],
    deps = [
        ":ifrt_service_proto_cc",
        "//xla/tsl/platform:status_to_from_proto",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "test_utils",
    srcs = ["test_utils.cc"],
    hdrs = ["test_utils.h"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/debugging:leak_check",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "prof_util",
    hdrs = ["prof_util.h"],
    deps = [
        ":ifrt_service_proto_cc",
        "//xla/tsl/profiler/utils:xplane_schema",
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/platform:random",
        "@local_tsl//tsl/profiler/lib:traceme",
        "@local_tsl//tsl/profiler/lib:traceme_encode",
    ],
)

cc_library(
    name = "versions",
    hdrs = ["versions.h"],
)

# copybara:uncomment_begin
# bzl_library(
#     name = "ifrt_proxy_bzl",
#     srcs = ["ifrt_proxy.bzl"],
#     parse_tests = False,
#     visibility = ["//visibility:private"],
#     deps = ["//xla/tsl:package_groups_bzl"],
# )
#
# bzl_library(
#     name = "ifrt_proxy_google_bzl",
#     srcs = ["ifrt_proxy.google.bzl"],
#     parse_tests = False,
#     visibility = ["//visibility:private"],
#     deps = [
#         "//devtools/build_cleaner/skylark:build_defs_lib",
#         "//xla:xla_bzl",
#     ],
# )
# copybara:uncomment_end

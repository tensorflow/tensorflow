load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/stream_executor:build_defs.bzl", "stream_executor_friends")
load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl/platform:build_config.bzl", "tf_proto_library")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([":friends"]),
    licenses = ["notice"],
)

package_group(
    name = "friends",
    packages = stream_executor_friends(),
)

cc_library(
    name = "executable_abi_version",
    srcs = ["executable_abi_version.cc"],
    hdrs = ["executable_abi_version.h"],
    deps = [
        ":executable_abi_version_proto_cc",
        "//xla/stream_executor:device_description",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

tf_proto_library(
    name = "executable_abi_version_proto",
    srcs = ["executable_abi_version.proto"],
)

xla_cc_test(
    name = "executable_abi_version_test",
    srcs = ["executable_abi_version_test.cc"],
    deps = [
        ":executable_abi_version",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:semantic_version",
        "//xla/stream_executor/cuda:cuda_compute_capability",
        "//xla/tsl/util/proto:proto_matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "runtime_abi_version",
    hdrs = ["runtime_abi_version.h"],
    deps = [
        ":executable_abi_version",
        ":runtime_abi_version_proto_cc",
        "//xla/stream_executor:platform_id",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

tf_proto_library(
    name = "runtime_abi_version_proto",
    srcs = ["runtime_abi_version.proto"],
)

cc_library(
    name = "runtime_abi_version_resolver",
    hdrs = ["runtime_abi_version_resolver.h"],
    deps = [
        ":runtime_abi_version",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "runtime_abi_version_manager",
    srcs = ["runtime_abi_version_manager.cc"],
    hdrs = ["runtime_abi_version_manager.h"],
    deps = [
        ":runtime_abi_version",
        ":runtime_abi_version_resolver",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
    ],
)

xla_cc_test(
    name = "runtime_abi_version_manager_test",
    srcs = ["runtime_abi_version_manager_test.cc"],
    deps = [
        ":runtime_abi_version_manager",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "mock_runtime_abi_version_resolver",
    testonly = True,
    hdrs = ["mock_runtime_abi_version_resolver.h"],
    deps = [
        ":runtime_abi_version",
        ":runtime_abi_version_resolver",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_for_library",
    ],
)

cc_library(
    name = "mock_runtime_abi_version",
    testonly = True,
    hdrs = ["mock_runtime_abi_version.h"],
    deps = [
        ":executable_abi_version",
        ":runtime_abi_version",
        "//xla/stream_executor:platform_id",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_for_library",
    ],
)

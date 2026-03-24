load("@rules_cc//cc:cc_library.bzl", "cc_library")

# Infrastructure for parallelization of compilation tasks.
load("//xla:xla.default.bzl", "xla_cc_test")
# copybara:uncomment load("//xla/tsl:tsl.default.bzl", "get_compatible_with_portable")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//xla:friends",
    ],
)

cc_library(
    name = "tsl_task_executor",
    srcs = ["tsl_task_executor.cc"],
    hdrs = ["tsl_task_executor.h"],
    # copybara:uncomment compatible_with = get_compatible_with_portable(),
    deps = [
        "//xla/tsl/concurrency:executor",
        "//xla/tsl/platform:env",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings:string_view",
        "@tsl//tsl/platform:platform_port",
    ],
)

cc_library(
    name = "concurrency_utils",
    srcs = ["concurrency_utils.cc"],
    hdrs = ["concurrency_utils.h"],
    # copybara:uncomment compatible_with = get_compatible_with_portable(),
    deps = [
        "//xla:future",
        "//xla/hlo/ir:hlo",
        "//xla/tsl/concurrency:executor",
        "//xla/tsl/platform:env",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:platform_port",
    ],
)

xla_cc_test(
    name = "concurrency_utils_test",
    size = "small",
    srcs = ["concurrency_utils_test.cc"],
    deps = [
        ":concurrency_utils",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/service:hlo_module_config",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

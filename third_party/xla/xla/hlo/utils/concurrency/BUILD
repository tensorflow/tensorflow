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
    name = "type_adapters",
    hdrs = ["type_adapters.h"],
    # copybara:uncomment compatible_with = get_compatible_with_portable(),
    deps = [
        "@com_google_absl//absl/functional:any_invocable",
    ],
)

cc_library(
    name = "tsl_task_executor",
    srcs = ["tsl_task_executor.cc"],
    hdrs = ["tsl_task_executor.h"],
    # copybara:uncomment compatible_with = get_compatible_with_portable(),
    deps = [
        ":type_adapters",
        "//xla/tsl/platform:env",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/synchronization",
        "@local_tsl//tsl/platform:platform_port",
    ],
)

## Tests below.

xla_cc_test(
    name = "type_adapters_test",
    srcs = ["type_adapters_test.cc"],
    deps = [
        ":type_adapters",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_cc_test(
    name = "tsl_task_executor_test",
    size = "small",
    srcs = ["tsl_task_executor_test.cc"],
    deps = [
        ":tsl_task_executor",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
    ],
)

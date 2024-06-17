load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")
load("//xla:xla.bzl", "xla_cc_test")

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
    name = "hlo_benchmark_runner",
    testonly = 1,
    srcs = ["hlo_benchmark_runner.cc"],
    hdrs = ["hlo_benchmark_runner.h"],
    deps = [
        "//xla:literal",
        "//xla/client:xla_computation",
        "//xla/hlo/ir:hlo",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_executable",
        "//xla/pjrt/cpu:cpu_client",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_parser",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test_benchmark",
    ],
)

xla_cc_test(
    name = "dag_execution_benchmark_test",
    srcs = ["dag_execution_benchmark_test.cc"],
    deps = [
        ":hlo_benchmark_runner",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:test_benchmark",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "dot_benchmark_test",
    srcs = ["dot_benchmark_test.cc"],
    deps = [
        ":hlo_benchmark_runner",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:test_benchmark",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "elementwise_benchmark_test",
    srcs = ["elementwise_benchmark_test.cc"],
    deps = [
        ":hlo_benchmark_runner",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:test_benchmark",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "fusion_benchmark_test",
    srcs = ["fusion_benchmark_test.cc"],
    deps = [
        ":hlo_benchmark_runner",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:test_benchmark",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "optimizer_benchmark_test",
    srcs = ["optimizer_benchmark_test.cc"],
    deps = [
        ":hlo_benchmark_runner",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:test_benchmark",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "reduction_benchmark_test",
    srcs = ["reduction_benchmark_test.cc"],
    deps = [
        ":hlo_benchmark_runner",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:test_benchmark",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "convolution_benchmark_test",
    srcs = ["convolution_benchmark_test.cc"],
    deps = [
        ":hlo_benchmark_runner",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:test_benchmark",
        "@local_tsl//tsl/platform:test_main",
    ],
)

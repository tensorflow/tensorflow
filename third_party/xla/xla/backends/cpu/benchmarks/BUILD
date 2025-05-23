load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/tsl:tsl.bzl", "tsl_copts")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

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
    name = "aot_benchmark_helper",
    testonly = 1,
    srcs = ["aot_benchmark_helper.cc"],
    hdrs = ["aot_benchmark_helper.h"],
    deps = [
        "//xla/service:compiler",
        "//xla/service/cpu:cpu_aot_compilation_result",
        "//xla/service/cpu:test_header_helper",
        "@com_google_absl//absl/strings:string_view",
    ],
)

xla_cc_test(
    name = "aot_benchmark_helper_test",
    srcs = ["aot_benchmark_helper_test.cc"],
    deps = [
        ":aot_benchmark_helper",
        "//xla/service:compiler",
        "//xla/service/cpu:cpu_aot_compilation_result",
        "//xla/service/cpu:test_header_helper",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "hlo_benchmark_runner",
    testonly = 1,
    srcs = ["hlo_benchmark_runner.cc"],
    hdrs = ["hlo_benchmark_runner.h"],
    deps = [
        "//xla:literal",
        "//xla:shape_util",
        "//xla:util",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_executable",
        "//xla/pjrt/cpu:cpu_client",
        "//xla/pjrt/plugin/xla_cpu:cpu_client_options",
        "//xla/pjrt/plugin/xla_cpu:xla_cpu_pjrt_client",
        "//xla/service:compiler",
        "//xla/service:hlo_module_config",
        "//xla/tests:test_utils",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test_benchmark",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:casts",
    ],
)

xla_cc_test(
    name = "aliasing_benchmark_test",
    srcs = ["aliasing_benchmark_test.cc"],
    deps = [
        ":hlo_benchmark_runner",
        ":multi_benchmark_config",
        "//xla:xla_data_proto_cc",
        "//xla/tsl/platform:test",
        "//xla/tsl/platform:test_benchmark",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:stacktrace_handler",
    ],
)

xla_cc_test(
    name = "dag_execution_benchmark_test",
    srcs = ["dag_execution_benchmark_test.cc"],
    fail_if_no_test_linked = False,  # NOLINT=This contains benchmarks only, no tests.
    deps = [
        ":hlo_benchmark_runner",
        ":multi_benchmark_config",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:test_benchmark",
        "//xla/tsl/platform:test_main",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "dot_benchmark_test",
    srcs = ["dot_benchmark_test.cc"],
    fail_if_no_test_linked = False,  # NOLINT=This contains benchmarks only, no tests.
    deps = [
        ":hlo_benchmark_runner",
        ":multi_benchmark_config",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/ir:hlo",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:test_benchmark",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:stacktrace_handler",
    ],
)

xla_cc_test(
    name = "dynamic_update_slice_benchmark_test",
    srcs = ["dynamic_update_slice_benchmark_test.cc"],
    fail_if_no_test_linked = False,  # NOLINT=This contains benchmarks only, no tests.
    deps = [
        ":hlo_benchmark_runner",
        ":multi_benchmark_config",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:test_benchmark",
        "//xla/tsl/platform:test_main",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "elementwise_benchmark_test",
    srcs = ["elementwise_benchmark_test.cc"],
    fail_if_no_test_linked = False,  # NOLINT=This contains benchmarks only, no tests.
    deps = [
        ":hlo_benchmark_runner",
        ":multi_benchmark_config",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:test_benchmark",
        "//xla/tsl/platform:test_main",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "exp_benchmark_test",
    srcs = ["exp_benchmark_test.cc"],
    fail_if_no_test_linked = False,  # NOLINT=This contains benchmarks only, no tests.
    deps = [
        ":hlo_benchmark_runner",
        ":multi_benchmark_config",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:test_benchmark",
        "//xla/tsl/platform:test_main",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "log_benchmark_test",
    srcs = ["log_benchmark_test.cc"],
    fail_if_no_test_linked = False,  # NOLINT=This contains benchmarks only, no tests.
    deps = [
        ":hlo_benchmark_runner",
        ":multi_benchmark_config",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:test_benchmark",
        "//xla/tsl/platform:test_main",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "fusion_benchmark_test",
    srcs = ["fusion_benchmark_test.cc"],
    fail_if_no_test_linked = False,  # NOLINT=This contains benchmarks only, no tests.
    deps = [
        ":hlo_benchmark_runner",
        ":multi_benchmark_config",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:test_benchmark",
        "//xla/tsl/platform:test_main",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "multi_benchmark_config",
    testonly = 1,
    hdrs = ["multi_benchmark_config.h"],
    deps = [
        ":aot_benchmark_helper",
        ":hlo_benchmark_runner",
        "//xla/tsl/platform:test_benchmark",
        "@com_google_absl//absl/strings:string_view",
    ],
)

xla_cc_test(
    name = "optimizer_benchmark_test",
    srcs = ["optimizer_benchmark_test.cc"],
    fail_if_no_test_linked = False,  # NOLINT=This contains benchmarks only, no tests.
    deps = [
        ":hlo_benchmark_runner",
        ":multi_benchmark_config",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:test_benchmark",
        "//xla/tsl/platform:test_main",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "reduction_benchmark_test",
    srcs = ["reduction_benchmark_test.cc"],
    fail_if_no_test_linked = False,  # NOLINT=This contains benchmarks only, no tests.
    deps = [
        ":hlo_benchmark_runner",
        ":multi_benchmark_config",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:test_benchmark",
        "//xla/tsl/platform:test_main",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "convolution_benchmark_test",
    srcs = ["convolution_benchmark_test.cc"],
    fail_if_no_test_linked = False,  # NOLINT=This contains benchmarks only, no tests.
    deps = [
        ":hlo_benchmark_runner",
        ":multi_benchmark_config",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:test_benchmark",
        "//xla/tsl/platform:test_main",
        "@com_google_absl//absl/strings",
    ],
)

xla_cc_test(
    name = "custom_call_benchmark_test",
    srcs = ["custom_call_benchmark_test.cc"],
    fail_if_no_test_linked = False,  # NOLINT=This contains benchmarks only, no tests.
    deps = [
        ":hlo_benchmark_runner",
        ":multi_benchmark_config",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/ffi",
        "//xla/ffi:ffi_api",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:test_benchmark",
        "//xla/tsl/platform:test_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "gather_benchmark_test",
    srcs = ["gather_benchmark_test.cc"],
    fail_if_no_test_linked = False,  # NOLINT=This contains benchmarks only, no tests.
    deps = [
        ":hlo_benchmark_runner",
        ":multi_benchmark_config",
        "//xla:array2d",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:test_benchmark",
        "//xla/tsl/platform:test_main",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
    ],
)

xla_cc_test(
    name = "select_and_scatter_benchmark_test",
    srcs = ["select_and_scatter_benchmark_test.cc"],
    fail_if_no_test_linked = False,  # NOLINT=This contains benchmarks only, no tests.
    deps = [
        ":hlo_benchmark_runner",
        ":multi_benchmark_config",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:test_benchmark",
        "//xla/tsl/platform:test_main",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "concatenate_benchmark_test",
    srcs = ["concatenate_benchmark_test.cc"],
    fail_if_no_test_linked = False,  # NOLINT=This contains benchmarks only, no tests.
    deps = [
        ":hlo_benchmark_runner",
        ":multi_benchmark_config",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:test_benchmark",
        "//xla/tsl/platform:test_main",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "topk_benchmark_test",
    srcs = ["topk_benchmark_test.cc"],
    fail_if_no_test_linked = False,  # NOLINT=This contains benchmarks only, no tests.
    deps = [
        ":hlo_benchmark_runner",
        ":multi_benchmark_config",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/tsl/platform:test_benchmark",
        "//xla/tsl/platform:test_main",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
    ],
)

xla_cc_test(
    name = "pad_benchmark_test",
    srcs = ["pad_benchmark_test.cc"],
    fail_if_no_test_linked = False,  # NOLINT=This contains benchmarks only, no tests.
    deps = [
        ":hlo_benchmark_runner",
        ":multi_benchmark_config",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:test_benchmark",
        "//xla/tsl/platform:test_main",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "onednn_matmul_benchmark_test",
    srcs = ["onednn_matmul_benchmark_test.cc"],
    copts = tsl_copts(),
    fail_if_no_test_linked = False,  # NOLINT=This contains benchmarks only, no tests.
    deps = [
        ":hlo_benchmark_runner",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/service/cpu:onednn_util",
        "//xla/tsl/platform:test",
        "//xla/tsl/platform:test_benchmark",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_cc_test(
    name = "tanh_benchmark_test",
    srcs = ["tanh_benchmark_test.cc"],
    fail_if_no_test_linked = False,  # NOLINT=This contains benchmarks only, no tests.
    deps = [
        ":hlo_benchmark_runner",
        ":multi_benchmark_config",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:test_benchmark",
        "//xla/tsl/platform:test_main",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "scatter_benchmark_test",
    srcs = ["scatter_benchmark_test.cc"],
    fail_if_no_test_linked = False,  # NOLINT=This contains benchmarks only, no tests.
    deps = [
        ":hlo_benchmark_runner",
        ":multi_benchmark_config",
        "//xla:array2d",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/service:scatter_simplifier",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:test_benchmark",
        "//xla/tsl/platform:test_main",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
    ],
)

xla_cc_test(
    name = "transposed_copy_benchmark_test",
    srcs = ["transposed_copy_benchmark_test.cc"],
    fail_if_no_test_linked = False,  # NOLINT=This contains benchmarks only, no tests.
    deps = [
        ":hlo_benchmark_runner",
        ":multi_benchmark_config",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:test_benchmark",
        "//xla/tsl/platform:test_main",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "transposed_dot_benchmark_test",
    srcs = ["transposed_dot_benchmark_test.cc"],
    fail_if_no_test_linked = False,  # NOLINT=This contains benchmarks only, no tests.
    deps = [
        ":hlo_benchmark_runner",
        ":multi_benchmark_config",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:test_benchmark",
        "//xla/tsl/platform:test_main",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "xnn_fusion_benchmark_test",
    srcs = ["xnn_fusion_benchmark_test.cc"],
    fail_if_no_test_linked = False,  # NOLINT=This contains benchmarks only, no tests.
    linkstatic = 1,  # required to override pthreadpool symbols
    deps = [
        ":hlo_benchmark_runner",
        ":multi_benchmark_config",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:test_benchmark",
        "//xla/tsl/platform:test_main",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

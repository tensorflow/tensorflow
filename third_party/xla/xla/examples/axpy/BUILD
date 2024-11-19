load("//xla:xla.bzl", "xla_cc_test")

# copybara:uncomment package(default_applicable_licenses = ["//tensorflow:license"])

xla_cc_test(
    name = "stablehlo_compile_test",
    srcs = ["stablehlo_compile_test.cc"],
    data = ["stablehlo_axpy.mlir"],
    deps = [
        "//xla:error_spec",
        "//xla:literal",
        "//xla:literal_util",
        "//xla/pjrt:pjrt_api",
        "//xla/pjrt:pjrt_c_api_client",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_executable",
        "//xla/pjrt/c:pjrt_c_api_cpu",
        "//xla/pjrt/c:pjrt_c_api_hdrs",
        "//xla/pjrt/plugin/xla_cpu:cpu_client_options",
        "//xla/pjrt/plugin/xla_cpu:xla_cpu_pjrt_client",
        "//xla/tests:literal_test_util",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:path",
        "@local_tsl//tsl/platform:status_matchers",
        "@local_tsl//tsl/platform:test",
        "@stablehlo//:register",
    ],
)

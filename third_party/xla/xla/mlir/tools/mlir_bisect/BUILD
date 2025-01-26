load("@bazel_skylib//rules:build_test.bzl", "build_test")
load("//xla:xla.bzl", "xla_cc_binary")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

# copybara:uncomment package(default_applicable_licenses = ["//tensorflow:license"])

build_test(
    name = "mlir-bisect_build_test",
    targets = [
        ":mlir-bisect",
    ],
)

xla_cc_binary(
    name = "mlir-bisect",
    testonly = True,
    srcs = ["mlir_bisect.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":bisect_lib",
        "//xla:literal",
        "//xla/mlir/tools/mlir_bisect/rewrites",
        "//xla/mlir/tools/mlir_interpreter/dialects",
        "//xla/mlir/tools/mlir_interpreter/framework",
        "//xla/mlir/tools/mlir_replay/public:execution_trace_proto_cc",
        "//xla/mlir/tools/mlir_replay/public:execution_trace_utils",
        "//xla/mlir_hlo:hlo_dialect_registration",
        "//xla/mlir_hlo:mhlo_passes",
        "//xla/service:hlo_proto_cc",
        "@com_google_absl//absl/log:check",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineUtils",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:platform_port",
    ],
)

cc_library(
    name = "bisect_lib",
    srcs = [
        "bisect_lib.cc",
        "test_passes.cc",
    ],
    hdrs = [
        "bisect_lib.h",
        "test_passes.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//xla/mlir/tools/mlir_replay/public:execution_trace_proto_cc",
        "//xla/mlir/tools/mlir_replay/public:execution_trace_proto_cc_impl",
        "//xla/mlir/tools/mlir_replay/public:execution_trace_utils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)

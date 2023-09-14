load("//xla:xla.bzl", "xla_cc_binary")
load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")
load("@bazel_skylib//rules:build_test.bzl", "build_test")

# copybara:uncomment package(default_applicable_licenses = ["//tensorflow:license"])

build_test(
    name = "mlir-bisect_build_test",
    targets = [
        ":mlir-bisect",
    ],
)

xla_cc_binary(
    name = "mlir-bisect",
    srcs = ["mlir_bisect.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":bisect_lib",
        "//xla/mlir/runtime/ir:rt",
        "//xla/mlir/tools/mlir_bisect/rewrites",
        "//xla/mlir/tools/mlir_replay/public:execution_trace_utils",
        "//xla/mlir_hlo:deallocation",
        "//xla/mlir_hlo:deallocation_passes",
        "//xla/mlir_hlo:gml_st",
        "//xla/mlir_hlo:gml_st_passes",
        "//xla/mlir_hlo:gml_st_test_passes",
        "//xla/mlir_hlo:hlo_dialect_registration",
        "//xla/mlir_hlo:lhlo",
        "//xla/mlir_hlo:lmhlo_passes",
        "//xla/mlir_hlo:mhlo_passes",
        "//xla/mlir_hlo:mlir_interpreter_dialects",
        "//xla/mlir_hlo:mlir_interpreter_framework",
        "//xla/mlir_hlo:thlo",
        "//xla/mlir_hlo:thlo_passes",
        "//xla/service:hlo_proto_cc",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MlirReduceLib",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
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
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)

load("//tensorflow/compiler/xla:xla.bzl", "xla_cc_binary")
load("//tensorflow/tsl/platform:rules_cc.bzl", "cc_library")
load("@bazel_skylib//rules:build_test.bzl", "build_test")

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
        "//tensorflow/compiler/xla/mlir/runtime/ir:rt",
        "//tensorflow/compiler/xla/mlir/tools/mlir_bisect/rewrites",
        "//tensorflow/compiler/xla/mlir/tools/mlir_replay/public:execution_trace_utils",
        "//tensorflow/compiler/xla/mlir_hlo:gml_st",
        "//tensorflow/compiler/xla/mlir_hlo:gml_st_passes",
        "//tensorflow/compiler/xla/mlir_hlo:gml_st_test_passes",
        "//tensorflow/compiler/xla/mlir_hlo:hlo_dialect_registration",
        "//tensorflow/compiler/xla/mlir_hlo:lhlo",
        "//tensorflow/compiler/xla/mlir_hlo:lmhlo_passes",
        "//tensorflow/compiler/xla/mlir_hlo:mhlo_passes",
        "//tensorflow/compiler/xla/mlir_hlo:mlir_interpreter_dialects",
        "//tensorflow/compiler/xla/mlir_hlo:mlir_interpreter_framework",
        "//tensorflow/compiler/xla/mlir_hlo:thlo",
        "//tensorflow/compiler/xla/mlir_hlo:thlo_passes",
        "//tensorflow/compiler/xla/service:hlo_proto_cc",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:platform_port",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MlirReduceLib",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
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
        "//tensorflow/compiler/xla/mlir/tools/mlir_replay/public:execution_trace_proto_cc",
        "//tensorflow/compiler/xla/mlir/tools/mlir_replay/public:execution_trace_proto_cc_impl",
        "//tensorflow/compiler/xla/mlir/tools/mlir_replay/public:execution_trace_utils",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)

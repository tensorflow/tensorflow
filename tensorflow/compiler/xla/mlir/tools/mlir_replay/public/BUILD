load("//tensorflow/tsl/platform:build_config.bzl", "tf_proto_library")
load("//tensorflow/tsl/platform:rules_cc.bzl", "cc_library")
load("//tensorflow/compiler/xla:xla.bzl", "xla_cc_test")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "compiler_trace_instrumentation",
    srcs = ["compiler_trace_instrumentation.cc"],
    hdrs = ["compiler_trace_instrumentation.h"],
    deps = [
        ":compiler_trace_proto_cc",
        ":compiler_trace_proto_cc_impl",
        "//tensorflow/compiler/xla/service/llvm_ir:llvm_util",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:path",
        "//tensorflow/tsl/platform:protobuf",
        "@com_google_absl//absl/strings:str_format",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
    ],
)

cc_library(
    name = "execution_trace_utils",
    srcs = ["execution_trace_utils.cc"],
    hdrs = ["execution_trace_utils.h"],
    deps = [
        ":execution_trace_proto_cc",
        ":execution_trace_proto_cc_impl",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/mlir_hlo:mlir_interpreter_framework",
        "//tensorflow/tsl/platform:statusor",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

xla_cc_test(
    name = "execution_trace_utils_test",
    srcs = ["execution_trace_utils_test.cc"],
    deps = [
        ":execution_trace_utils",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla/mlir_hlo:mlir_interpreter_framework",
        "//tensorflow/tsl/platform:statusor",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Support",
    ],
)

tf_proto_library(
    name = "execution_trace_proto",
    srcs = ["execution_trace.proto"],
    cc_api_version = 2,
    make_default_target_header_only = True,
)

tf_proto_library(
    name = "compiler_trace_proto",
    srcs = ["compiler_trace.proto"],
    cc_api_version = 2,
    make_default_target_header_only = True,
)

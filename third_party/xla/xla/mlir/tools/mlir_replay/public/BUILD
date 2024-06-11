load("@local_tsl//tsl/platform:build_config.bzl", "tf_proto_library")
load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")
load("//xla:xla.bzl", "xla_cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "compiler_trace_instrumentation",
    srcs = ["compiler_trace_instrumentation.cc"],
    hdrs = ["compiler_trace_instrumentation.h"],
    deps = [
        ":compiler_trace_proto_cc",
        ":compiler_trace_proto_cc_impl",
        "//xla/service/llvm_ir:llvm_util",
        "@com_google_absl//absl/strings:str_format",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:path",
        "@local_tsl//tsl/platform:protobuf",
    ],
)

cc_library(
    name = "execution_trace_utils",
    srcs = ["execution_trace_utils.cc"],
    hdrs = ["execution_trace_utils.h"],
    deps = [
        ":execution_trace_proto_cc",
        ":execution_trace_proto_cc_impl",
        "//xla:literal",
        "//xla:xla_data_proto_cc",
        "//xla/mlir/tools/mlir_interpreter/framework",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "execution_trace_utils_test",
    srcs = ["execution_trace_utils_test.cc"],
    deps = [
        ":execution_trace_utils",
        "//xla:literal",
        "//xla:literal_util",
        "//xla/mlir/tools/mlir_interpreter/framework",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Support",
        "@local_tsl//tsl/platform:statusor",
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

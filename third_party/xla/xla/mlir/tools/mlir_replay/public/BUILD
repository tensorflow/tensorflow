load("@local_tsl//tsl/platform:build_config.bzl", "tf_proto_library")
load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")

package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "compiler_trace_instrumentation",
    srcs = ["compiler_trace_instrumentation.cc"],
    hdrs = ["compiler_trace_instrumentation.h"],
    visibility = ["//visibility:public"],
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

tf_proto_library(
    name = "compiler_trace_proto",
    srcs = ["compiler_trace.proto"],
    cc_api_version = 2,
    make_default_target_header_only = True,
    visibility = ["//visibility:public"],
)

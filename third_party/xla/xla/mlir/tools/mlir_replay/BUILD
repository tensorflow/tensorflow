load("//xla:xla.bzl", "xla_cc_binary")
load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")
load("@bazel_skylib//rules:build_test.bzl", "build_test")

# copybara:uncomment package(default_applicable_licenses = ["//tensorflow:license"])

build_test(
    name = "mlir_replay_build_test",
    targets = [
        ":mlir_replay",
    ],
)

xla_cc_binary(
    name = "mlir_replay",
    srcs = ["mlir_replay.cc"],
    deps = [
        ":mlir_replay_lib",
        "//xla:debug_options_flags",
        "//xla/mlir/framework/ir:xla_framework",
        "//xla/mlir/runtime/ir:rt",
        "//xla/mlir/tools/mlir_replay/public:compiler_trace_proto_cc",
        "//xla/mlir/tools/mlir_replay/public:compiler_trace_proto_cc_impl",
        "//xla/mlir/tools/mlir_replay/public:execution_trace_proto_cc",
        "//xla/mlir/tools/mlir_replay/public:execution_trace_utils",
        "//xla/mlir/xla_cpu/ir:xla_cpu",
        "//xla/mlir_hlo:deallocation",
        "//xla/mlir_hlo:gml_st",
        "//xla/mlir_hlo:hlo_dialect_registration",
        "//xla/mlir_hlo:lhlo",
        "//xla/mlir_hlo:lhlo_gpu",
        "//xla/mlir_hlo:mlir_interpreter_dialects",
        "//xla/mlir_hlo:mlir_interpreter_framework",
        "//xla/mlir_hlo:thlo",
        "//xla/service:hlo_proto_cc",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:IR",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:path",
        "@local_tsl//tsl/platform:platform_port",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/util:command_line_flags",
    ],
)

cc_library(
    name = "mlir_replay_lib",
    srcs = ["mlir_replay_lib.cc"],
    hdrs = ["mlir_replay_lib.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//xla:xla_data_proto_cc",
        "//xla/mlir/framework/ir:xla_framework",
        "//xla/mlir/tools/mlir_replay/public:execution_trace_proto_cc",
        "//xla/mlir/tools/mlir_replay/public:execution_trace_utils",
        "//xla/mlir_hlo:mlir_interpreter_framework",
        "//xla/service:hlo_proto_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/random:bit_gen_ref",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MlirReduceLib",
        "@llvm-project//mlir:Support",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
    ],
)

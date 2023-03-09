load("//tensorflow/compiler/xla:xla.bzl", "xla_cc_binary")
load("//tensorflow/tsl/platform:rules_cc.bzl", "cc_library")
load("@bazel_skylib//rules:build_test.bzl", "build_test")

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
        "//tensorflow/compiler/xla:debug_options_flags",
        "//tensorflow/compiler/xla/mlir/runtime/ir:rt",
        "//tensorflow/compiler/xla/mlir/tools/mlir_replay/public:compiler_trace_proto_cc",
        "//tensorflow/compiler/xla/mlir/tools/mlir_replay/public:compiler_trace_proto_cc_impl",
        "//tensorflow/compiler/xla/mlir/tools/mlir_replay/public:execution_trace_proto_cc",
        "//tensorflow/compiler/xla/mlir/tools/mlir_replay/public:execution_trace_utils",
        "//tensorflow/compiler/xla/mlir/xla_cpu/ir:xla_cpu",
        "//tensorflow/compiler/xla/mlir_hlo:deallocation",
        "//tensorflow/compiler/xla/mlir_hlo:gml_st",
        "//tensorflow/compiler/xla/mlir_hlo:hlo_dialect_registration",
        "//tensorflow/compiler/xla/mlir_hlo:lhlo",
        "//tensorflow/compiler/xla/mlir_hlo:lhlo_gpu",
        "//tensorflow/compiler/xla/mlir_hlo:mlir_interpreter_dialects",
        "//tensorflow/compiler/xla/mlir_hlo:mlir_interpreter_framework",
        "//tensorflow/compiler/xla/mlir_hlo:thlo",
        "//tensorflow/compiler/xla/service:hlo_proto_cc",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:path",
        "//tensorflow/tsl/platform:platform_port",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/util:command_line_flags",
        "@com_google_absl//absl/strings:str_format",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "mlir_replay_lib",
    srcs = ["mlir_replay_lib.cc"],
    hdrs = ["mlir_replay_lib.h"],
    deps = [
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/mlir/tools/mlir_replay/public:execution_trace_proto_cc",
        "//tensorflow/compiler/xla/mlir/tools/mlir_replay/public:execution_trace_utils",
        "//tensorflow/compiler/xla/mlir_hlo:mlir_interpreter_framework",
        "//tensorflow/compiler/xla/service:hlo_proto_cc",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:statusor",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/random:bit_gen_ref",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MlirReduceLib",
        "@llvm-project//mlir:Support",
    ],
)

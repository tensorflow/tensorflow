load("@bazel_skylib//rules:build_test.bzl", "build_test")
load("//xla:xla.default.bzl", "xla_cc_binary")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

# copybara:uncomment package(default_applicable_licenses = ["//tensorflow:license"])

build_test(
    name = "mlir_replay_build_test",
    targets = [
        ":mlir_replay",
    ],
)

xla_cc_binary(
    name = "mlir_replay",
    testonly = True,
    srcs = ["mlir_replay.cc"],
    deps = [
        ":mlir_replay_lib",
        "//xla:debug_options_flags",
        "//xla/mlir/tools/mlir_interpreter/dialects",
        "//xla/mlir/tools/mlir_interpreter/framework",
        "//xla/mlir/tools/mlir_replay/public:compiler_trace_proto_cc",
        "//xla/mlir/tools/mlir_replay/public:compiler_trace_proto_cc_impl",
        "//xla/mlir/tools/mlir_replay/public:execution_trace_proto_cc",
        "//xla/mlir/tools/mlir_replay/public:execution_trace_utils",
        "//xla/mlir_hlo:hlo_dialect_registration",
        "//xla/service:hlo_proto_cc",
        "//xla/tsl/util:command_line_flags",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:IR",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:path",
        "@local_tsl//tsl/platform:platform_port",
        "@local_tsl//tsl/platform:status",
    ],
)

cc_library(
    name = "mlir_replay_lib",
    srcs = ["mlir_replay_lib.cc"],
    hdrs = ["mlir_replay_lib.h"],
    deps = [
        "//xla/mlir/tools/mlir_interpreter/framework",
        "//xla/mlir/tools/mlir_replay/public:execution_trace_proto_cc",
        "//xla/mlir/tools/mlir_replay/public:execution_trace_utils",
        "//xla/service:hlo_proto_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/random:bit_gen_ref",
        "@com_google_absl//absl/random:distributions",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:ParseUtilities",
        "@llvm-project//mlir:Support",
        "@local_tsl//tsl/platform:statusor",
    ],
)

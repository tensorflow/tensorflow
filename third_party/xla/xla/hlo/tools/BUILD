# Tools and utilities that aid in XLA development and usage.

load("//xla:py_strict.bzl", "py_strict_library")
load(
    "//xla:xla.default.bzl",
    "xla_cc_binary",
)
load("//xla/tsl:tsl.default.bzl", "filegroup")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//xla:internal"],
    licenses = ["notice"],
)

# Filegroup used to collect source files for dependency checking.
filegroup(
    name = "c_srcs",
    data = glob([
        "**/*.cc",
        "**/*.h",
    ]),
    visibility = ["//xla:internal"],
)

xla_cc_binary(
    name = "hex_floats_to_packed_literal",
    srcs = ["hex_floats_to_packed_literal.cc"],
    deps = [
        "//xla/tsl/lib/io:buffered_inputstream",
        "//xla/tsl/lib/io:random_inputstream",
        "//xla/tsl/util:command_line_flags",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:platform_port",
        "@local_tsl//tsl/platform:status",
    ],
)

xla_cc_binary(
    name = "show_literal",
    srcs = ["show_literal.cc"],
    deps = [
        "//xla:literal",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:platform_port",
        "@local_tsl//tsl/platform:status",
    ],
)

xla_cc_binary(
    name = "convert_computation",
    srcs = ["convert_computation.cc"],
    deps = [
        "//xla/service:hlo_proto_cc",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:platform_port",
        "@local_tsl//tsl/platform:protobuf",
        "@local_tsl//tsl/platform:status",
    ],
)

xla_cc_binary(
    name = "hlo_module_metadata_processor",
    srcs = ["hlo_module_metadata_processor.cc"],
    deps = [
        "//xla/service:hlo_proto_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:platform_port",
        "@local_tsl//tsl/platform:protobuf",
        "@local_tsl//tsl/platform:status",
    ],
)

xla_cc_binary(
    name = "show_text_literal",
    srcs = ["show_text_literal.cc"],
    deps = [
        "//xla:literal",
        "//xla:text_literal_reader",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:platform_port",
        "@local_tsl//tsl/platform:protobuf",
    ],
)

xla_cc_binary(
    name = "hlo_proto_to_json",
    srcs = ["hlo_proto_to_json.cc"],
    deps = [
        "//xla:util",
        "//xla/service:hlo_proto_cc",
        "//xla/tsl/util:command_line_flags",
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:platform_port",
        "@local_tsl//tsl/platform:status",
    ],
)

xla_cc_binary(
    name = "hlo-translate",
    testonly = True,
    srcs = ["hlo_translate.cc"],
    deps = [
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/translate:stablehlo",
        "//xla/hlo/translate/hlo_to_mhlo:hlo_to_mlir_hlo",
        "//xla/mlir_hlo:mhlo_passes",
        "//xla/service:hlo_proto_cc",
        "//xla/service:hlo_proto_util",
        "//xla/service/cpu:cpu_compiler",
        "//xla/service/cpu:cpu_transfer_manager",
        "//xla/service/llvm_ir:llvm_util",
        "//xla/stream_executor/host:host_platform",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TranslateLib",
        "@local_tsl//tsl/platform:protobuf",
        "@stablehlo//:stablehlo_passes",
    ],
)

xla_cc_binary(
    name = "hlo-opt",
    testonly = True,
    linkopts = ["-Wl,-rpath,$$ORIGIN/../lit_lib"],
    deps = [
        "//xla/hlo/tools/hlo_opt:opt_main",
    ],
)

py_strict_library(
    name = "generate_hlo_test_checks",
    srcs = ["generate_hlo_test_checks.py"],
    srcs_version = "PY3",
)

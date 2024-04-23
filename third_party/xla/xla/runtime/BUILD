load("@local_tsl//tsl/platform:build_config.bzl", "tf_platform_deps")
load(
    "@local_tsl//tsl/platform:build_config_root.bzl",
    "if_llvm_aarch32_available",
    "if_llvm_aarch64_available",
    "if_llvm_powerpc_available",
    "if_llvm_system_z_available",
    "if_llvm_x86_available",
)
load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")
load("//xla:xla.bzl", "xla_cc_test")
load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl:tsl.default.bzl", "filegroup", "get_compatible_with_portable")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility(["//xla:internal"]),
    licenses = ["notice"],
)

cc_library(
    name = "arguments",
    srcs = ["arguments.cc"],
    hdrs = ["arguments.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":async_runtime",
        ":types",
        "//xla:shape_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@llvm-project//llvm:Support",
    ],
)

xla_cc_test(
    name = "arguments_test",
    srcs = ["arguments_test.cc"],
    deps = [
        ":arguments",
        "@local_tsl//tsl/platform:test_benchmark",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "async_runtime",
    srcs = ["async_runtime.cc"],
    hdrs = ["async_runtime.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "@com_google_absl//absl/base:dynamic_annotations",
        "@local_tsl//tsl/concurrency:async_value",
        "@local_tsl//tsl/concurrency:ref_count",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:platform_port",
    ],
)

xla_cc_test(
    name = "async_runtime_test",
    srcs = ["async_runtime_test.cc"],
    deps = [
        ":async_runtime",
        "@local_tsl//tsl/concurrency:async_value",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "async_values_cache",
    hdrs = ["async_values_cache.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "@local_tsl//tsl/platform",
    ] + tf_platform_deps(
        "async_values_cache",
        platform_dir = "//xla/runtime/",
    ),
)

cc_library(
    name = "constraints",
    srcs = ["constraints.cc"],
    hdrs = ["constraints.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
    ],
)

filegroup(
    name = "aot_ffi_execution_context_hdrs",
    srcs = ["aot_ffi_execution_context.h"],
)

cc_library(
    name = "aot_ffi_execution_context",
    hdrs = ["aot_ffi_execution_context.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "aot_ffi",
    srcs = ["aot_ffi.cc"],
    hdrs = ["aot_ffi.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":aot_ffi_c_symbols",
        "//xla/runtime/ffi:ffi_api",
    ],
)

cc_library(
    name = "aot_ffi_c_symbols",
    srcs = ["aot_ffi_c_symbols.cc"],
    hdrs = ["aot_ffi_c_symbols.h"],
    visibility = ["//visibility:public"],
    deps = [":aot_ffi_execution_context"],
)

cc_library(
    name = "custom_call",
    srcs = ["custom_call.cc"],
    hdrs = ["custom_call.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":async_runtime",
        ":diagnostics",
        ":errors",
        ":logical_result",
        ":map_by_type",
        ":memref_view",
        ":state",
        ":type_id",
        "//xla:shape_util",
        "//xla/runtime/ffi:ffi_abi",
        "@com_google_absl//absl/base:dynamic_annotations",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@eigen_archive//:eigen3",
        "@llvm-project//llvm:Support",
        "@local_tsl//tsl/concurrency:async_value",
    ],
)

cc_library(
    name = "custom_call_registry",
    srcs = ["custom_call_registry.cc"],
    hdrs = ["custom_call_registry.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":custom_call",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "diagnostics",
    srcs = ["diagnostics.cc"],
    hdrs = ["diagnostics.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":logical_result",
        "@com_google_absl//absl/status",
        "@llvm-project//llvm:Support",
        "@local_tsl//tsl/platform:logging",
    ],
)

xla_cc_test(
    name = "diagnostics_test",
    srcs = ["diagnostics_test.cc"],
    deps = [
        ":diagnostics",
        "@com_google_absl//absl/status",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "errors",
    hdrs = ["errors.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "executable",
    srcs = ["executable.cc"],
    hdrs = ["executable.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":arguments",
        ":async_runtime",
        ":custom_call",
        ":custom_call_registry",
        ":diagnostics",
        ":errors",
        ":execution_engine",
        ":logical_result",
        ":memory_mapper",
        ":results",
        ":runtime",
        ":type_id",
        ":types",
        "//xla/mlir/runtime/utils:async_runtime_api",
        "//xla/mlir/runtime/utils:c_runner_utils",
        "//xla/mlir/runtime/utils:float_16bits",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@llvm-project//llvm:OrcJIT",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "execution_engine",
    srcs = ["execution_engine.cc"],
    hdrs = ["execution_engine.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":errors",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:ExecutionEngine",
        "@llvm-project//llvm:OrcJIT",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:Target",
        "@llvm-project//llvm:TransformUtils",
    ] + if_llvm_aarch32_available([
        "@llvm-project//llvm:ARMAsmParser",
        "@llvm-project//llvm:ARMCodeGen",
    ]) + if_llvm_aarch64_available([
        "@llvm-project//llvm:AArch64AsmParser",
        "@llvm-project//llvm:AArch64CodeGen",
    ]) + if_llvm_powerpc_available([
        "@llvm-project//llvm:PowerPCAsmParser",
        "@llvm-project//llvm:PowerPCCodeGen",
    ]) + if_llvm_system_z_available([
        "@llvm-project//llvm:SystemZAsmParser",
        "@llvm-project//llvm:SystemZCodeGen",
    ]) + if_llvm_x86_available([
        "@llvm-project//llvm:X86AsmParser",
        "@llvm-project//llvm:X86CodeGen",
    ]),
)

cc_library(
    name = "jit_executable",
    srcs = ["jit_executable.cc"],
    hdrs = ["jit_executable.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":async_values_cache",
        ":constraints",
        ":errors",
        "//xla/mlir/runtime/transforms:jit_compiler",
        "//xla/mlir/runtime/utils:constraints",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@llvm-project//llvm:Support",
        "@local_tsl//tsl/concurrency:async_value",
    ],
)

cc_library(
    name = "logical_result",
    hdrs = ["logical_result.h"],
    compatible_with = get_compatible_with_portable(),
    deps = ["@llvm-project//mlir:Support"],
)

cc_library(
    name = "map_by_type",
    hdrs = ["map_by_type.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":type_id",
        "@llvm-project//llvm:Support",
    ],
)

xla_cc_test(
    name = "map_by_type_test",
    srcs = ["map_by_type_test.cc"],
    deps = [
        ":map_by_type",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_benchmark",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "memory_mapper",
    srcs = ["memory_mapper.cc"],
    hdrs = ["memory_mapper.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "@llvm-project//llvm:ExecutionEngine",
        "@llvm-project//llvm:Support",
        "@local_tsl//tsl/platform",
    ] + tf_platform_deps(
        "memory_mapper",
        platform_dir = "//xla/runtime/",
    ),
)

cc_library(
    name = "memref_view",
    hdrs = ["memref_view.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//xla:xla_data_proto_cc",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "module",
    hdrs = ["module.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":custom_call_registry",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "module_registry",
    srcs = ["module_registry.cc"],
    hdrs = ["module_registry.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":module",
    ],
)

xla_cc_test(
    name = "module_test",
    srcs = ["module_test.cc"],
    deps = [
        ":custom_call",
        ":module",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "results",
    hdrs = ["results.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":logical_result",
        ":types",
    ],
)

xla_cc_test(
    name = "results_test",
    srcs = ["results_test.cc"],
    deps = [
        ":logical_result",
        ":results",
        ":types",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_benchmark",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "runtime",
    hdrs = ["runtime.h"],
    compatible_with = get_compatible_with_portable(),
)

cc_library(
    name = "state",
    hdrs = ["state.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
    ],
)

xla_cc_test(
    name = "state_test",
    srcs = ["state_test.cc"],
    deps = [
        ":state",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_benchmark",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "symbolic_shape",
    srcs = ["symbolic_shape.cc"],
    hdrs = ["symbolic_shape.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":arguments",
        ":constraints",
        ":logical_result",
        ":types",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@llvm-project//llvm:Support",
    ],
)

xla_cc_test(
    name = "symbolic_shape_test",
    srcs = ["symbolic_shape_test.cc"],
    deps = [
        ":arguments",
        ":constraints",
        ":symbolic_shape",
        ":types",
        "@llvm-project//llvm:Support",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_benchmark",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "types",
    srcs = ["types.cc"],
    hdrs = ["types.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "tracing",
    hdrs = ["tracing.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":custom_call",
        ":type_id",
    ],
)

cc_library(
    name = "type_id",
    srcs = ["type_id.cc"],
    hdrs = ["type_id.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "compiler",
    hdrs = ["compiler.h"],
    compatible_with = get_compatible_with_portable(),
)

cc_library(
    name = "cpu_event",
    hdrs = ["cpu_event.h"],
)

xla_cc_test(
    name = "type_id_test",
    srcs = ["type_id_test.cc"],
    deps = [
        ":type_id",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_benchmark",
        "@local_tsl//tsl/platform:test_main",
    ],
)

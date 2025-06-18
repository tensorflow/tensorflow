load("//xla:py_strict.bzl", "py_strict_test")
load("//xla:xla.default.bzl", "xla_cc_test", "xla_internal")
load("//xla/tsl:tsl.bzl", "internal_visibility")
load(
    "//xla/tsl/platform:build_config_root.bzl",
    "if_llvm_aarch64_available",
    "if_llvm_powerpc_available",
    "if_llvm_system_z_available",
    "if_llvm_x86_available",
)
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility(["//xla/backends/cpu:xla_backend_cpu_internal_access"]),
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//xla:friends",
    ],
)

cc_library(
    name = "contiguous_section_memory_manager",
    srcs = ["contiguous_section_memory_manager.cc"],
    hdrs = ["contiguous_section_memory_manager.h"],
    deps = [
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@llvm-project//llvm:ExecutionEngine",
        "@llvm-project//llvm:Support",
        # TODO(basioli): This dependency increases the binary size significantly.
        # Consider reducing the dependency size, or use something alternative.
        "//xla:util",
    ],
)

cc_library(
    name = "cpu_features",
    srcs = ["cpu_features.cc"],
    hdrs = ["cpu_features.h"],
    deps = [
        "//xla/tsl/platform:logging",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TargetParser",
        "@local_tsl//tsl/platform:platform_port",
    ],
)

cc_library(
    name = "ir_compiler",
    srcs = ["ir_compiler.cc"],
    hdrs = ["ir_compiler.h"],
    deps = [
        ":cpu_features",
        ":polynomial_approximations",
        "//xla:util",
        "//xla:xla_proto_cc",
        "//xla/service:hlo_module_config",
        "//xla/service/cpu:cpu_options",
        "//xla/service/llvm_ir:llvm_util",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@llvm-project//llvm:Analysis",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:ExecutionEngine",
        "@llvm-project//llvm:Instrumentation",
        "@llvm-project//llvm:MC",
        "@llvm-project//llvm:Object",
        "@llvm-project//llvm:OrcJIT",
        "@llvm-project//llvm:Passes",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:Target",
        "@llvm-project//llvm:TargetParser",
        "@local_tsl//tsl/platform:platform_port",
    ],
)

cc_library(
    name = "fusion_compiler",
    srcs = ["fusion_compiler.cc"],
    hdrs = ["fusion_compiler.h"],
    deps = [
        "//xla:status_macros",
        "//xla:util",
        "//xla/backends/cpu/codegen/emitters/ir:xla_cpu",
        "//xla/backends/cpu/codegen/emitters/transforms:passes",
        "//xla/codegen:llvm_ir_kernel_source",
        "//xla/codegen:mlir_kernel_source",
        "//xla/codegen/emitters/ir:xla",
        "//xla/codegen/emitters/ir:xla_attrs_inc_gen",
        "//xla/codegen/emitters/transforms:passes",
        "//xla/mlir/tools/mlir_replay/public:compiler_trace_proto_cc",
        "//xla/mlir_hlo",
        "//xla/mlir_hlo:mhlo_passes",
        "//xla/tsl/framework/mlir:status_scoped_diagnostic_handler",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Core",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:AffineToStandard",
        "@llvm-project//mlir:AffineTransforms",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:BuiltinToLLVMIRTranslation",
        "@llvm-project//mlir:ComplexToStandard",
        "@llvm-project//mlir:ControlFlowDialect",
        "@llvm-project//mlir:DLTIDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncExtensions",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:LLVMIRTransforms",
        "@llvm-project//mlir:LLVMToLLVMIRTranslation",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:MemRefTransforms",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ReconcileUnrealizedCasts",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:SCFToControlFlow",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:ToLLVMIRTranslation",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:VectorDialect",
    ],
)

cc_library(
    name = "jit_compiler",
    srcs = ["jit_compiler.cc"],
    hdrs = ["jit_compiler.h"],
    deps = [
        ":compiled_function_library",
        ":contiguous_section_memory_manager",
        ":cpu_features",
        ":execution_engine",
        ":ir_compiler",
        ":object_loader",
        "//xla:util",
        "//xla/backends/cpu/runtime:function_library",
        "//xla/service/cpu:orc_jit_memory_mapper",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:ExecutionEngine",
        "@llvm-project//llvm:MC",
        "@llvm-project//llvm:OrcJIT",
        "@llvm-project//llvm:OrcShared",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:Target",
        "@llvm-project//llvm:TargetParser",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:platform_port",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/profiler/lib:traceme",
        "@local_tsl//tsl/profiler/lib:traceme_encode",
    ] + if_llvm_aarch64_available([
        "@llvm-project//llvm:AArch64CodeGen",  # fixdeps: keep
    ]) + if_llvm_powerpc_available([
        "@llvm-project//llvm:PowerPCCodeGen",  # fixdeps: keep
    ]) + if_llvm_system_z_available([
        "@llvm-project//llvm:SystemZCodeGen",  # fixdeps: keep
    ]) + if_llvm_x86_available([
        "@llvm-project//llvm:X86CodeGen",  # fixdeps: keep
    ]) + xla_internal(["service/cpu:named_orc_jit_memory_mapper"]),
)

xla_cc_test(
    name = "jit_compiler_test",
    srcs = ["jit_compiler_test.cc"],
    deps = [
        ":ir_compiler",
        ":jit_compiler",
        "//xla:util",
        "//xla/backends/cpu/runtime:function_library",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:AsmParser",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:OrcJIT",
        "@llvm-project//llvm:OrcShared",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:Target",
    ],
)

cc_library(
    name = "polynomial_approximations",
    srcs = ["polynomial_approximations.cc"],
    hdrs = ["polynomial_approximations.h"],
    deps = [
        ":vector_ir_builder",
        "//xla:xla_data_proto_cc",
        "//xla/service/llvm_ir:llvm_util",
        "//xla/service/llvm_ir:math_ops",
        "//xla/tsl/platform:logging",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//llvm:Analysis",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TransformUtils",
    ],
)

cc_library(
    name = "target_machine_features",
    srcs = ["target_machine_features.cc"],
    hdrs = ["target_machine_features.h"],
    deps = [
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/backends/cpu:alignment",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Analysis",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:Target",
        "@llvm-project//llvm:ir_headers",
    ],
)

cc_library(
    name = "target_machine_test_base",
    testonly = True,
    hdrs = ["target_machine_test_base.h"],
    deps = [
        ":target_machine_features",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_for_library",
        "@llvm-project//llvm:ARMCodeGen",  # fixdeps: keep
        "@llvm-project//llvm:ExecutionEngine",
        "@llvm-project//llvm:MC",
        "@llvm-project//llvm:Target",
        "@llvm-project//llvm:TargetParser",
        "@llvm-project//llvm:X86CodeGen",  # fixdeps: keep
    ],
)

xla_cc_test(
    name = "target_machine_features_test",
    srcs = ["target_machine_features_test.cc"],
    deps = [
        ":target_machine_features",
        ":target_machine_test_base",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "vector_ir_builder",
    srcs = ["vector_ir_builder.cc"],
    hdrs = ["vector_ir_builder.h"],
    deps = [
        ":target_machine_features",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/service/llvm_ir:llvm_util",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "kernel_api_ir_builder",
    srcs = ["kernel_api_ir_builder.cc"],
    hdrs = ["kernel_api_ir_builder.h"],
    deps = [
        ":symbol_name_util",
        "//xla:cpu_function_runtime",
        "//xla:shape_util",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/service:buffer_assignment",
        "//xla/service:hlo_module_config",
        "//xla/service/llvm_ir:ir_array",
        "//xla/service/llvm_ir:llvm_util",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:ir_headers",
    ],
)

xla_cc_test(
    name = "kernel_api_ir_builder_test",
    srcs = ["kernel_api_ir_builder_test.cc"],
    deps = [
        ":kernel_api_ir_builder",
        "//xla:cpu_function_runtime",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/analysis:hlo_ordering",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:filecheck",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/service:buffer_assignment",
        "//xla/service:buffer_value",
        "//xla/service:hlo_module_config",
        "//xla/service:logical_buffer",
        "//xla/service/cpu:cpu_executable",
        "//xla/service/llvm_ir:ir_array",
        "//xla/service/llvm_ir:llvm_util",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:ir_headers",
    ],
)

cc_library(
    name = "compiled_function_library",
    srcs = ["compiled_function_library.cc"],
    hdrs = ["compiled_function_library.h"],
    deps = [
        ":execution_engine",
        "//xla/backends/cpu/runtime:function_library",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

xla_cc_test(
    name = "aot_compiled_function_library_test",
    srcs = ["aot_compiled_function_library_test.cc"],
    deps = [
        ":aot_compiled_function_library",
        "//xla/backends/cpu/runtime:function_library",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "aot_compiled_function_library",
    srcs = ["aot_compiled_function_library.cc"],
    hdrs = ["aot_compiled_function_library.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//xla/backends/cpu/runtime:function_library",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "object_loader",
    srcs = ["object_loader.cc"],
    hdrs = ["object_loader.h"],
    deps = [
        ":compiled_function_library",
        ":contiguous_section_memory_manager",
        ":execution_engine",
        "//xla/backends/cpu/runtime:function_library",
        "//xla/service/cpu:orc_jit_memory_mapper",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:JITLink",
        "@llvm-project//llvm:OrcShared",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:ir_headers",
    ],
)

cc_library(
    name = "execution_engine",
    srcs = ["execution_engine.cc"],
    hdrs = ["execution_engine.h"],
    deps = [
        ":contiguous_section_memory_manager",
        "//xla/service/cpu:orc_jit_memory_mapper",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@llvm-project//llvm:ExecutionEngine",
        "@llvm-project//llvm:JITLink",
        "@llvm-project//llvm:OrcJIT",  # buildcleaner: keep
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:ir_headers",
    ],
)

cc_library(
    name = "computation_kernel_emitter",
    srcs = ["computation_kernel_emitter.cc"],
    hdrs = ["computation_kernel_emitter.h"],
    deps = [
        ":kernel_api_ir_builder",
        ":target_machine_features",
        "//xla:shape_util",
        "//xla:util",
        "//xla/codegen:kernel_definition",
        "//xla/codegen:kernel_emitter",
        "//xla/codegen:kernel_spec",
        "//xla/codegen:llvm_ir_kernel_source",
        "//xla/hlo/ir:hlo",
        "//xla/service:buffer_assignment",
        "//xla/service/cpu:backend_config_proto_cc",
        "//xla/service/cpu:ir_emitter",
        "//xla/service/llvm_ir:ir_array",
        "//xla/service/llvm_ir:llvm_util",
        "//xla/stream_executor:launch_dim",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:ir_headers",
    ],
)

py_strict_test(
    name = "scatter_kernel_emitter_test",
    srcs = ["scatter_kernel_emitter_test.py"],
    tags = [
        "no_oss",
    ],
    deps = [
        "//third_party/py/numpy",
        "//xla/backends/cpu/testlib",
        "//xla/codegen/testlib",
        "//xla/python:xla_extension",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)

xla_cc_test(
    name = "object_loader_test",
    srcs = ["object_loader_test.cc"],
    deps = [
        ":execution_engine",
        ":ir_compiler",
        ":jit_compiler",
        ":object_loader",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/backends/cpu/runtime:function_library",
        "//xla/service:cpu_plugin",
        "//xla/service/cpu:executable_proto_cc",
        "//xla/service/llvm_ir:llvm_util",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:AsmParser",
        "@llvm-project//llvm:JITLink",
        "@llvm-project//llvm:Object",
        "@llvm-project//llvm:OrcShared",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:Target",
        "@llvm-project//llvm:ir_headers",
    ],
)

cc_library(
    name = "symbol_name_util",
    srcs = ["symbol_name_util.cc"],
    hdrs = ["symbol_name_util.h"],
    deps = [
        "//xla:util",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

xla_cc_test(
    name = "symbol_name_util_test",
    srcs = ["symbol_name_util_test.cc"],
    deps = [
        ":symbol_name_util",
        "//xla/tsl/platform:statusor",
        "@com_google_googletest//:gtest_main",
    ],
)

py_strict_test(
    name = "computation_kernel_emitter_test",
    srcs = ["computation_kernel_emitter_test.py"],
    tags = [
        "no_oss",
    ],
    deps = [
        "//third_party/py/numpy",
        "//xla/backends/cpu/testlib",
        "//xla/codegen/testlib",
        "//xla/python:xla_extension",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)

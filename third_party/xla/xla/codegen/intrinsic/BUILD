load("//xla:xla.default.bzl", "xla_cc_test")
load(
    "//xla/tsl/platform:build_config_root.bzl",
    "if_llvm_aarch64_available",
    "if_llvm_powerpc_available",
    "if_llvm_riscv_available",
    "if_llvm_system_z_available",
    "if_llvm_x86_available",
)
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//xla:friends",
    ],
)

cc_library(
    name = "intrinsic",
    hdrs = ["intrinsic.h"],
    deps = [
        ":type",
        ":vec_name_mangler",
        "//xla:xla_data_proto_cc",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Target",
        "@llvm-project//llvm:ir_headers",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
    ],
)

xla_cc_test(
    name = "intrinsic_test",
    srcs = ["intrinsic_test.cc"],
    deps = [
        ":type",
        "//xla:xla_data_proto_cc",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "fptrunc",
    srcs = ["fptrunc.cc"],
    hdrs = ["fptrunc.h"],
    deps = [
        ":intrinsic",
        ":type",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/service/llvm_ir:llvm_util",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:ir_headers",
    ],
)

xla_cc_test(
    name = "fptrunc_test",
    srcs = ["fptrunc_test.cc"],
    deps = [
        ":fptrunc",
        ":simple_jit_runner",
        "//xla:xla_data_proto_cc",
        "//xla/codegen/intrinsic:type",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:Target",
        "@llvm-project//llvm:ir_headers",
    ],
)

cc_library(
    name = "ldexp",
    srcs = ["ldexp.cc"],
    hdrs = ["ldexp.h"],
    deps = [
        ":intrinsic",
        "//xla:xla_data_proto_cc",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Core",  # buildcleaner: keep
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:ir_headers",
    ],
)

xla_cc_test(
    name = "ldexp_test",
    srcs = ["ldexp_test.cc"],
    deps = [
        ":intrinsic",
        ":ldexp",
        ":simple_jit_runner",
        ":test_matchers",
        "//xla:xla_data_proto_cc",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:JITLink",
        "@llvm-project//llvm:ir_headers",
    ],
)

cc_library(
    name = "simple_jit_runner",
    testonly = True,
    srcs = ["simple_jit_runner.cc"],
    hdrs = ["simple_jit_runner.h"],
    deps = [
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/service/llvm_ir:llvm_util",
        "//xla/tsl/util:safe_reinterpret_cast",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:dynamic_annotations",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@llvm-project//llvm:ExecutionEngine",
        "@llvm-project//llvm:JITLink",
        "@llvm-project//llvm:MC",
        "@llvm-project//llvm:OrcJIT",  # buildcleaner: keep
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:Target",
        "@llvm-project//llvm:TargetParser",
        "@llvm-project//llvm:ir_headers",
    ] + if_llvm_aarch64_available([
        "@llvm-project//llvm:AArch64AsmParser",  # fixdeps: keep
        "@llvm-project//llvm:AArch64CodeGen",  # fixdeps: keep
    ]) + if_llvm_powerpc_available([
        "@llvm-project//llvm:PowerPCAsmParser",  # fixdeps: keep
        "@llvm-project//llvm:PowerPCCodeGen",  # fixdeps: keep
    ]) + if_llvm_riscv_available([
        "@llvm-project//llvm:RISCVAsmParser",  # fixdeps: keep
        "@llvm-project//llvm:RISCVCodeGen",  # fixdeps: keep
    ]) + if_llvm_system_z_available([
        "@llvm-project//llvm:SystemZAsmParser",  # fixdeps: keep
        "@llvm-project//llvm:SystemZCodeGen",  # fixdeps: keep
    ]) + if_llvm_x86_available([
        "@llvm-project//llvm:X86AsmParser",  # fixdeps: keep
        "@llvm-project//llvm:X86CodeGen",  # fixdeps: keep
    ]),
)

xla_cc_test(
    name = "simple_jit_runner_test",
    srcs = ["simple_jit_runner_test.cc"],
    deps = [
        ":simple_jit_runner",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:ir_headers",
    ],
)

cc_library(
    name = "test_matchers",
    testonly = True,
    srcs = [],
    hdrs = ["test_matchers.h"],
    deps = [
        "//xla:fp_util",
        "@com_google_googletest//:gtest_for_library",
    ],
)

cc_library(
    name = "vec_name_mangler",
    hdrs = ["vec_name_mangler.h"],
    deps = [
        ":type",
        "//xla:util",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "vec_name_mangler_test",
    srcs = ["vec_name_mangler_test.cc"],
    deps = [
        ":vec_name_mangler",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "string_interner",
    hdrs = ["string_interner.h"],
    deps = ["@com_google_absl//absl/strings:string_view"],
)

cc_library(
    name = "intrinsic_compiler_lib",
    srcs = ["intrinsic_compiler_lib.cc"],
    hdrs = ["intrinsic_compiler_lib.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//llvm:Analysis",
        "@llvm-project//llvm:IPO",
        "@llvm-project//llvm:InstCombine",
        "@llvm-project//llvm:Passes",
        "@llvm-project//llvm:Scalar",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TransformUtils",
        "@llvm-project//llvm:ir_headers",
    ],
)

xla_cc_test(
    name = "intrinsic_compiler_lib_test",
    srcs = ["intrinsic_compiler_lib_test.cc"],
    deps = [
        ":intrinsic_compiler_lib",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:ir_headers",
    ],
)

cc_library(
    name = "exp",
    srcs = ["exp.cc"],
    hdrs = ["exp.h"],
    deps = [
        ":intrinsic",
        ":ldexp",
        "//xla:xla_data_proto_cc",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Analysis",
        "@llvm-project//llvm:Core",  # buildcleaner: keep
        "@llvm-project//llvm:ExecutionEngine",
        "@llvm-project//llvm:IPO",
        "@llvm-project//llvm:JITLink",
        "@llvm-project//llvm:Passes",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:Target",
        "@llvm-project//llvm:TransformUtils",
        "@llvm-project//llvm:ir_headers",
    ],
)

cc_library(
    name = "log1p",
    srcs = ["log1p.cc"],
    hdrs = ["log1p.h"],
    deps = [
        ":intrinsic",
        "//xla:xla_data_proto_cc",
        "//xla/service/llvm_ir:llvm_util",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:ir_headers",
    ],
)

xla_cc_test(
    name = "log1p_test",
    srcs = ["log1p_test.cc"],
    deps = [
        ":intrinsic",
        ":log1p",
        ":simple_jit_runner",
        ":test_matchers",
        "//xla:xla_data_proto_cc",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:JITLink",
        "@llvm-project//llvm:ir_headers",
    ],
)

xla_cc_test(
    name = "exp_test",
    srcs = ["exp_test.cc"],
    deps = [
        ":exp",
        ":intrinsic",
        ":ldexp",
        ":simple_jit_runner",
        ":test_matchers",
        "//xla:xla_data_proto_cc",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:JITLink",
        "@llvm-project//llvm:ir_headers",
    ],
)

cc_library(
    name = "rsqrt",
    srcs = ["rsqrt.cc"],
    hdrs = ["rsqrt.h"],
    deps = [
        ":intrinsic",
        "//xla:xla_data_proto_cc",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Core",  # buildcleaner: keep
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:Target",
        "@llvm-project//llvm:ir_headers",
    ],
)

cc_library(
    name = "erf",
    srcs = ["erf.cc"],
    hdrs = ["erf.h"],
    deps = [
        ":intrinsic",
        "//xla:xla_data_proto_cc",
        "//xla/service/llvm_ir:llvm_util",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:ir_headers",
    ],
)

xla_cc_test(
    name = "erf_test",
    srcs = ["erf_test.cc"],
    deps = [
        ":erf",
        ":intrinsic",
        ":simple_jit_runner",
        ":test_matchers",
        "//xla:xla_data_proto_cc",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:JITLink",
        "@llvm-project//llvm:ir_headers",
    ],
)

xla_cc_test(
    name = "rsqrt_test",
    srcs = ["rsqrt_test.cc"],
    deps = [
        ":intrinsic",
        ":rsqrt",
        ":simple_jit_runner",
        ":test_matchers",
        ":type",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/log",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:JITLink",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TargetParser",
        "@llvm-project//llvm:ir_headers",
    ],
)

xla_cc_test(
    name = "rsqrt_benchmark_test",
    srcs = ["rsqrt_benchmark_test.cc"],
    fail_if_no_test_linked = False,  # NOLINT=This contains benchmarks only, no tests.
    fail_if_no_test_selected = False,  # NOLINT=This contains benchmarks only, no tests.
    deps = [
        ":intrinsic",
        ":rsqrt",
        ":simple_jit_runner",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/tsl/platform:test_benchmark",
        "//xla/tsl/platform:test_main",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Target",
        "@llvm-project//llvm:ir_headers",
    ],
)

cc_library(
    name = "tanh",
    srcs = ["tanh.cc"],
    hdrs = ["tanh.h"],
    deps = [
        ":intrinsic",
        "//xla:xla_data_proto_cc",
        "//xla/service/llvm_ir:llvm_util",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:ir_headers",
    ],
)

xla_cc_test(
    name = "tanh_test",
    srcs = ["tanh_test.cc"],
    deps = [
        ":intrinsic",
        ":simple_jit_runner",
        ":tanh",
        ":test_matchers",
        "//xla:xla_data_proto_cc",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:JITLink",
        "@llvm-project//llvm:ir_headers",
    ],
)

cc_library(
    name = "type",
    srcs = ["type.cc"],
    hdrs = ["type.h"],
    deps = [
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/mlir/utils:type_util",
        "//xla/service/llvm_ir:llvm_util",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:ir_headers",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

xla_cc_test(
    name = "type_test",
    srcs = ["type_test.cc"],
    deps = [
        ":type",
        "//xla:xla_data_proto_cc",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:ir_headers",
    ],
)

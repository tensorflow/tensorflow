load("//xla:xla.bzl", "xla_cc_test")
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
    name = "contiguous_section_memory_manager",
    srcs = ["contiguous_section_memory_manager.cc"],
    hdrs = ["contiguous_section_memory_manager.h"],
    deps = [
        "//xla:util",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:ExecutionEngine",
        "@llvm-project//llvm:OrcJIT",
        "@llvm-project//llvm:Support",
        "@local_tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "function_library",
    hdrs = ["function_library.h"],
    deps = [
        "//xla:util",
        "//xla/backends/cpu/runtime:kernel_c_api",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "jit_compiler",
    srcs = ["jit_compiler.cc"],
    hdrs = ["jit_compiler.h"],
    deps = [
        ":function_library",
        "//xla:util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@llvm-project//llvm:OrcJIT",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:Target",
    ],
)

xla_cc_test(
    name = "jit_compiler_test",
    srcs = ["jit_compiler_test.cc"],
    deps = [
        ":jit_compiler",
        "@llvm-project//llvm:AsmParser",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:OrcJIT",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:Target",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
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
        "@llvm-project//llvm:Analysis",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TransformUtils",
        "@local_tsl//tsl/platform:logging",
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
        "@llvm-project//llvm:Analysis",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:Target",
        "@llvm-project//llvm:ir_headers",
        "@local_tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "vector_ir_builder",
    srcs = ["vector_ir_builder.cc"],
    hdrs = ["vector_ir_builder.h"],
    deps = [
        ":target_machine_features",
        "//xla:shape_util",
        "//xla:types",
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

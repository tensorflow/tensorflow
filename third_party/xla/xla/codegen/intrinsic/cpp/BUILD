load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/tsl/platform:rules_cc.bzl", "cc_binary", "cc_library")
load(":cc_to_llvm_ir.bzl", "cc_ir_header")

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

# This empty target provides the C++ compilation context for the C++ source file.
cc_library(
    name = "cpp_context_provider",
    srcs = [],
)

cc_binary(
    name = "embed_bitcode",
    srcs = ["embed_bitcode.cc"],
    deps = [
        "//tensorflow/compiler/aot:embedded_constant_buffers",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:AArch64AsmParser",  # buildcleaner: keep
        "@llvm-project//llvm:AArch64CodeGen",  # buildcleaner: keep
        "@llvm-project//llvm:ARMAsmParser",  # buildcleaner: keep
        "@llvm-project//llvm:ARMCodeGen",  # buildcleaner: keep
        "@llvm-project//llvm:Object",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:X86AsmParser",  # buildcleaner: keep
        "@llvm-project//llvm:X86CodeGen",  # buildcleaner: keep
    ],
)

cc_library(
    name = "tanh",
    srcs = ["tanh.cc"],
    hdrs = ["tanh.h"],
    copts = [
        "-Wno-psabi",
    ],
    deps = [
        ":vector_ops",
    ],
)

cc_library(
    name = "vector_ops",
    hdrs = ["vector_ops.h"],
    deps = [
        "@eigen_archive//:eigen3",
    ],
)

cc_ir_header(
    name = "tanh_ll",
    src = "tanh.cc",
    deps = [
        ":cpp_context_provider",
        ":tanh",
        ":vector_ops",
    ],
)

xla_cc_test(
    name = "tanh_test",
    srcs = ["tanh_test.cc"],
    deps = [
        ":_cpp_gen_intrinsics",
        ":tanh_ll",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:IRReader",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "eigen_unary_lib",
    srcs = ["eigen_unary.cc"],
    hdrs = ["eigen_unary.h"],
    copts = [
        "-Wno-psabi",
    ],
    deps = [
        ":vector_ops",
        "@eigen_archive//:eigen3",
    ],
)

cc_ir_header(
    name = "eigen_unary_ll",
    src = "eigen_unary.cc",
    deps = [
        ":cpp_context_provider",
        ":eigen_unary_lib",
        ":vector_ops",
        "@eigen_archive//:eigen3",
    ],
)

xla_cc_test(
    name = "eigen_unary_test",
    srcs = ["eigen_unary_test.cc"],
    copts = [
        "-Wno-psabi",
    ],
    deps = [
        ":_cpp_gen_intrinsics",
        ":eigen_unary_lib",
        ":eigen_unary_ll",
        ":vector_ops",
        "//xla/codegen/intrinsic:test_matchers",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:IRReader",
        "@llvm-project//llvm:Support",
    ],
)

cc_binary(
    name = "dump_llvm_ir",
    srcs = ["dump_llvm_ir.cc"],
    deps = [
        ":_cpp_gen_intrinsics",
        ":eigen_unary_ll",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:IRReader",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "_cpp_gen_intrinsics",
    srcs = ["cpp_gen_intrinsics.cc"],
    hdrs = ["cpp_gen_intrinsics.h"],
    deps = [
        ":eigen_unary_ll",
        "//xla/codegen/intrinsic",
        "//xla/service/llvm_ir:llvm_util",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//llvm:IRReader",
        "@llvm-project//llvm:Linker",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:Target",
        "@llvm-project//llvm:TransformUtils",
        "@llvm-project//llvm:ir_headers",
    ],
)

cc_library(
    name = "_intrinsic_declarations",
    hdrs = ["intrinsic_declarations.h"],
    deps = [
        ":_cpp_gen_intrinsics",
        "//xla:xla_data_proto_cc",
        "//xla/codegen/intrinsic",
        "//xla/codegen/intrinsic:type",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//llvm:ir_headers",
    ],
)

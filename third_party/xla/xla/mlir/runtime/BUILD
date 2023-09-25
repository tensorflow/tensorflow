load("//xla:xla.bzl", "xla_cc_binary")
load("@local_tsl//tsl:tsl.default.bzl", "get_compatible_with_portable")
load("@bazel_skylib//rules:build_test.bzl", "build_test")

package_group(
    name = "friends",
    packages = [
        # copybara:uncomment_begin(google-only)
        # "//platforms/xla/service/cpu/...",
        # "//third_party/mlir_edge/tpgen/...",
        # # TODO(ezhulenev): Clean up dependencies that are leftovers from Autofusion project.
        # "@tf_runtime//...",
        # "//third_party/tf_runtime_google/...",
        # copybara:uncomment_end(google-only)
        "//tensorflow/compiler/mlir/tfrt/...",
        "//xla/mlir/...",
        "//xla/runtime/...",
        # TODO(ezhulenev): All targets depending on mlir must be under xla/mlir folder
        "//xla/service/cpu/...",
        "//xla/service/gpu/...",
    ],
)

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

build_test(
    name = "xla-runtime-opt_build_test",
    targets = [
        ":xla-runtime-opt",
    ],
)

xla_cc_binary(
    name = "xla-runtime-opt",
    srcs = ["xla-runtime-opt.cc"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//xla/mlir/math/transforms:passes",
        "//xla/mlir/memref/transforms:passes",
        "//xla/mlir/runtime/ir/tests:testlib",
        "//xla/mlir/runtime/transforms:compilation_pipeline_cpu",
        "//xla/mlir/runtime/transforms:compilation_pipeline_gpu",
        "//xla/mlir/runtime/transforms:passes",
        "@llvm-project//mlir:AsyncDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncExtensions",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:MlirOptLib",
    ],
)

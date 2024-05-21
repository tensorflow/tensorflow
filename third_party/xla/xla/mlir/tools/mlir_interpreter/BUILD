load("@bazel_skylib//rules:build_test.bzl", "build_test")
load("//xla:xla.bzl", "xla_cc_binary")

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

build_test(
    name = "mlir-interpreter-runner_build_test",
    targets = [
        ":mlir-interpreter-runner",
    ],
)

xla_cc_binary(
    name = "mlir-interpreter-runner",
    testonly = True,
    srcs = ["mlir_interpreter_runner.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//xla/mlir/tools/mlir_interpreter/dialects",
        "//xla/mlir/tools/mlir_interpreter/framework",
        "//xla/mlir_hlo:hlo_dialect_registration",
        "//xla/mlir_hlo:mhlo_passes",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:ParseUtilities",
        "@llvm-project//mlir:Support",
    ],
)

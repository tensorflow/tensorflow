load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    default_visibility = [
        ":friends",
    ],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = ["@llvm-project//mlir:subpackages"],
    packages = [
        "//tensorflow/compiler/mlir/...",
        "//tensorflow/compiler/mlir/lite/...",
    ],
)

cc_library(
    name = "tf_to_quant",
    srcs = [
        "tf_to_quant.cc",
    ],
    hdrs = [
        "passes.h",
    ],
    deps = [
        "//tensorflow/compiler/mlir/lite/quantization:quantization_lib",
        "//tensorflow/compiler/mlir/tensorflow",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:TransformUtils",
    ],
    alwayslink = 1,
)

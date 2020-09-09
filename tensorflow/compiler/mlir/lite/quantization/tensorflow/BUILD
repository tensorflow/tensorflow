package(
    default_visibility = [
        ":friends",
    ],
    licenses = ["notice"],  # Apache 2.0
)

package_group(
    name = "friends",
    includes = ["//third_party/mlir:subpackages"],
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
        "//tensorflow/compiler/mlir/lite/quantization:quantization_config",
        "//tensorflow/compiler/mlir/lite/quantization:quantization_lib",
        "//tensorflow/compiler/mlir/tensorflow",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:QuantOps",
    ],
    alwayslink = 1,
)

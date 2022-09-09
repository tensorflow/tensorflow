load("//tensorflow:tensorflow.bzl", "if_google")

package(
    licenses = ["notice"],
)

cc_library(
    name = "pass_utils",
    srcs = ["pass_utils.cc"],
    hdrs = ["pass_utils.h"],
    tags = ["gpu"],
    visibility = if_google([
        "//platforms/xla/tests/gpu:__pkg__",
    ]) + [
        "//tensorflow/compiler/mlir/tfrt:__pkg__",
        "//tensorflow/compiler/xla/service/gpu:__pkg__",
    ],
    deps = [
        "//tensorflow/compiler/mlir/tensorflow:dump_mlir_util",
        "//tensorflow/compiler/xla/mlir/transforms/gpu:passes",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
    ],
)

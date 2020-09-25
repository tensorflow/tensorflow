load("//tensorflow:tensorflow.bzl", "tf_gen_op_wrapper_py")

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
        "//learning/brain/experimental/mlir/tfr/...",
        "//tensorflow/compiler/mlir/...",
    ],
)

filegroup(
    name = "decomposition_lib",
    srcs = ["decomposition_lib.mlir"],
)

cc_library(
    name = "composite_ops_cc",
    srcs = ["composite_ops.cc"],
    copts = [
        "-Wno-unused-result",
        "-Wno-unused-variable",
    ],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ],
    alwayslink = 1,
)

tf_gen_op_wrapper_py(
    name = "composite_ops",
    out = "composite_ops.py",
    deps = [
        ":composite_ops_cc",
    ],
)

cc_library(
    name = "test_ops_cc",
    testonly = 1,
    srcs = ["test_ops.cc"],
    copts = [
        "-Wno-unused-result",
        "-Wno-unused-variable",
    ],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ],
    alwayslink = 1,
)

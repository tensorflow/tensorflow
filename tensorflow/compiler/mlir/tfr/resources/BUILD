load("//tensorflow:tensorflow.bzl", "tf_custom_op_library", "tf_gen_op_wrapper_py")
load("//tensorflow:tensorflow.bzl", "tf_custom_op_py_library")

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
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ],
    alwayslink = 1,
)

tf_custom_op_library(
    name = "composite_ops.so",
    srcs = [
        "composite_ops.cc",
    ],
)

tf_gen_op_wrapper_py(
    name = "gen_composite_ops",
    out = "gen_composite_ops.py",
    deps = [
        ":composite_ops_cc",
    ],
)

tf_custom_op_py_library(
    name = "composite_ops",
    dso = [":composite_ops.so"],
    kernels = [":composite_ops_cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":gen_composite_ops",
    ],
)

cc_library(
    name = "test_ops_cc",
    srcs = ["test_ops.cc"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ],
    alwayslink = 1,
)

tf_custom_op_library(
    name = "test_ops.so",
    srcs = [
        "test_ops.cc",
    ],
)

tf_gen_op_wrapper_py(
    name = "gen_test_ops",
    out = "gen_test_ops.py",
    deps = [
        ":test_ops_cc",
    ],
)

tf_custom_op_py_library(
    name = "test_ops",
    dso = ["test_ops.so"],
    kernels = [
        ":test_ops_cc",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":gen_test_ops",
    ],
)

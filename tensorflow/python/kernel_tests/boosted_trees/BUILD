# Description:
#   Kernel tests for Boosted Trees.

load("//tensorflow:tensorflow.bzl", "tf_py_test")

package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)

tf_py_test(
    name = "resource_ops_test",
    size = "small",
    srcs = ["resource_ops_test.py"],
    tags = [
        "no_mac",  # b/173427076
    ],
    deps = [
        "//tensorflow/core/kernels/boosted_trees:boosted_trees_proto_py",
        "//tensorflow/python:boosted_trees_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:resources",
        "//tensorflow/python:training",
        "//tensorflow/python:variables",
    ],
)

tf_py_test(
    name = "prediction_ops_test",
    size = "small",
    srcs = ["prediction_ops_test.py"],
    tags = [
        "no_mac",  # b/173427076
    ],
    deps = [
        "//tensorflow/core/kernels/boosted_trees:boosted_trees_proto_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:boosted_trees_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:resources",
    ],
)

tf_py_test(
    name = "stats_ops_test",
    size = "medium",
    srcs = ["stats_ops_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:boosted_trees_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "training_ops_test",
    size = "small",
    srcs = ["training_ops_test.py"],
    tags = [
        "no_mac",  # b/173427076
    ],
    deps = [
        "//tensorflow/core/kernels/boosted_trees:boosted_trees_proto_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:boosted_trees_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:resources",
    ],
)

tf_py_test(
    name = "quantile_ops_test",
    size = "small",
    srcs = ["quantile_ops_test.py"],
    deps = [
        "//tensorflow/core/kernels/boosted_trees:boosted_trees_proto_py",
        "//tensorflow/python:boosted_trees_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:resources",
    ],
)

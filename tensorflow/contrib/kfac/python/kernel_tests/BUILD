package(default_visibility = ["//visibility:private"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "py_test")

py_test(
    name = "estimator_test",
    srcs = ["estimator_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/kfac/python/ops:fisher_estimator",
        "//tensorflow/contrib/kfac/python/ops:layer_collection",
        "//tensorflow/contrib/kfac/python/ops:utils",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:variable_scope",
    ],
)

py_test(
    name = "fisher_factors_test",
    srcs = ["fisher_factors_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/kfac/python/ops:fisher_factors",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:gradients",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_seed",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "fisher_blocks_test",
    srcs = ["fisher_blocks_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/kfac/python/ops:fisher_blocks",
        "//tensorflow/contrib/kfac/python/ops:layer_collection",
        "//tensorflow/contrib/kfac/python/ops:utils",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:random_seed",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "layer_collection_test",
    srcs = ["layer_collection_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/kfac/python/ops:fisher_factors",
        "//tensorflow/contrib/kfac/python/ops:layer_collection",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:linalg_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:random_seed",
        "//tensorflow/python:variable_scope",
    ],
)

py_test(
    name = "optimizer_test",
    srcs = ["optimizer_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/kfac/python/ops:kfac_optimizer",
        "//tensorflow/contrib/kfac/python/ops:layer_collection",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "utils_test",
    srcs = ["utils_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/kfac/python/ops:utils",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:linalg_ops",
        "//tensorflow/python:random_seed",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "op_queue_test",
    srcs = ["op_queue_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/kfac/python/ops:op_queue",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
    ],
)

py_test(
    name = "loss_functions_test",
    srcs = ["loss_functions_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/kfac/python/ops:loss_functions",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:random_ops",
        "//third_party/py/numpy",
    ],
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

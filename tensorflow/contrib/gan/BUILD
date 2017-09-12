# Files for using TFGAN framework.
package(default_visibility = ["//tensorflow:__subpackages__"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "py_test")

py_library(
    name = "gan",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":features",
        ":losses",
    ],
)

py_library(
    name = "losses",
    srcs = ["python/losses/__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":losses_impl",
        ":tuple_losses",
        "//tensorflow/python:util",
    ],
)

py_library(
    name = "features",
    srcs = ["python/features/__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":clip_weights",
        ":conditioning_utils",
        ":virtual_batchnorm",
        "//tensorflow/python:util",
    ],
)

py_library(
    name = "losses_impl",
    srcs = ["python/losses/python/losses_impl.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/framework:framework_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:clip_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:gradients",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:summary",
        "//tensorflow/python:tensor_util",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/ops/distributions",
        "//tensorflow/python/ops/losses",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "losses_impl_test",
    srcs = ["python/losses/python/losses_impl_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":losses_impl",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:clip_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:random_seed",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/ops/distributions",
        "//tensorflow/python/ops/losses",
    ],
)

py_library(
    name = "tuple_losses",
    srcs = [
        "python/losses/python/losses_wargs.py",
        "python/losses/python/tuple_losses.py",
        "python/losses/python/tuple_losses_impl.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":losses_impl",
        "//tensorflow/python:util",
    ],
)

py_test(
    name = "tuple_losses_test",
    srcs = ["python/losses/python/tuple_losses_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":tuple_losses",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "conditioning_utils",
    srcs = [
        "python/features/python/conditioning_utils.py",
        "python/features/python/conditioning_utils_impl.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/layers:layers_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:embedding_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:tensor_util",
        "//tensorflow/python:variable_scope",
    ],
)

py_test(
    name = "conditioning_utils_test",
    srcs = ["python/features/python/conditioning_utils_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":conditioning_utils",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
    ],
)

py_library(
    name = "virtual_batchnorm",
    srcs = [
        "python/features/python/virtual_batchnorm.py",
        "python/features/python/virtual_batchnorm_impl.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:tensor_util",
        "//tensorflow/python:variable_scope",
    ],
)

py_test(
    name = "virtual_batchnorm_test",
    srcs = ["python/features/python/virtual_batchnorm_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":virtual_batchnorm",
        "//tensorflow/contrib/framework:framework_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:layers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:random_seed",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "clip_weights",
    srcs = [
        "python/features/python/clip_weights.py",
        "python/features/python/clip_weights_impl.py",
    ],
    srcs_version = "PY2AND3",
    deps = ["//tensorflow/contrib/opt:opt_py"],
)

py_test(
    name = "clip_weights_test",
    srcs = ["python/features/python/clip_weights_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":clip_weights",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:training",
        "//tensorflow/python:variables",
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

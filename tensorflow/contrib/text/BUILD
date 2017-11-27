# Description:
#   contains parts of TensorFlow that are experimental or unstable and which
#   are not supported.

package(default_visibility = [
    "//learning/brain:__subpackages__",
    "//tensorflow:__subpackages__",
])

licenses(["notice"])  # Apache 2.0

load(
    "//tensorflow:tensorflow.bzl",
    "py_test",
    "tf_custom_op_library",
    "tf_custom_op_py_library",
    "tf_gen_op_libs",
    "tf_gen_op_wrapper_py",
    "tf_kernel_library",
)

tf_custom_op_py_library(
    name = "text_py",
    srcs = [
        "__init__.py",
        "python/ops/__init__.py",
        "python/ops/skip_gram_ops.py",
    ],
    dso = [
        ":python/ops/_skip_gram_ops.so",
    ],
    kernels = [
        ":all_kernels",
        ":all_ops",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":gen_skip_gram_ops",
        "//tensorflow/contrib/lookup:lookup_py",
        "//tensorflow/contrib/util:util_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:check_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:random_seed",
        "//tensorflow/python:training",
        "//tensorflow/python:util",
    ],
)

tf_kernel_library(
    name = "skip_gram_kernels",
    srcs = ["kernels/skip_gram_kernels.cc"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//third_party/eigen3",
    ],
    alwayslink = 1,
)

cc_library(
    name = "all_kernels",
    deps = [":skip_gram_kernels"],
)

tf_custom_op_library(
    name = "python/ops/_skip_gram_ops.so",
    srcs = [
        "kernels/skip_gram_kernels.cc",
        "ops/skip_gram_ops.cc",
    ],
)

tf_gen_op_libs(
    op_lib_names = ["skip_gram_ops"],
)

cc_library(
    name = "all_ops",
    deps = [":skip_gram_ops_op_lib"],
)

tf_gen_op_wrapper_py(
    name = "gen_skip_gram_ops",
    out = "python/ops/gen_skip_gram_ops.py",
    deps = [":skip_gram_ops_op_lib"],
)

py_test(
    name = "skip_gram_ops_test",
    size = "medium",
    srcs = ["python/ops/skip_gram_ops_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":text_py",
        "//tensorflow/contrib/lookup:lookup_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:lookup_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_seed",
        "//tensorflow/python:training",
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
)

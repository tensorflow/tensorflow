# Description:
#   contains parts of TensorFlow that are experimental or unstable and which are not supported.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

load("//tensorflow:tensorflow.bzl", "cuda_py_test")
load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")
load("//tensorflow:tensorflow.bzl", "tf_gen_op_wrapper_py")
load("//tensorflow:tensorflow.bzl", "tf_gen_op_libs")

py_library(
    name = "framework_py",
    srcs = [
        "__init__.py",
        "python/framework/__init__.py",
        "python/framework/checkpoint_utils.py",
        "python/framework/deprecation.py",
        "python/framework/tensor_util.py",
        "python/ops/__init__.py",
        "python/ops/arg_scope.py",
        "python/ops/ops.py",
        "python/ops/prettyprint_ops.py",
        "python/ops/variables.py",
    ],
    data = [
        ":python/ops/_variable_ops.so",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":gen_variable_ops",
    ],
)

tf_custom_op_library(
    name = "python/ops/_variable_ops.so",
    srcs = [
        "kernels/zero_initializer_op.cc",
        "kernels/zero_initializer_op.h",
        "ops/variable_ops.cc",
    ],
)

tf_gen_op_libs(
    op_lib_names = ["variable_ops"],
)

cc_library(
    name = "all_ops",
    deps = [
        ":variable_ops_op_lib",
    ],
)

tf_gen_op_wrapper_py(
    name = "gen_variable_ops",
    out = "python/ops/gen_variable_ops.py",
    deps = [
        ":variable_ops_op_lib",
    ],
)

py_test(
    name = "arg_scope_test",
    size = "small",
    srcs = ["python/ops/arg_scope_test.py"],
    srcs_version = "PY2AND3",
    deps = ["//tensorflow:tensorflow_py"],
)

py_test(
    name = "checkpoint_utils_test",
    size = "small",
    srcs = ["python/framework/checkpoint_utils_test.py"],
    srcs_version = "PY2AND3",
    tags = ["manual"],  # http://b/30468735
    deps = ["//tensorflow:tensorflow_py"],
)

py_test(
    name = "ops_test",
    size = "small",
    srcs = ["python/ops/ops_test.py"],
    srcs_version = "PY2AND3",
    deps = ["//tensorflow:tensorflow_py"],
)

py_test(
    name = "prettyprint_ops_test",
    size = "small",
    srcs = ["python/ops/prettyprint_ops_test.py"],
    srcs_version = "PY2AND3",
    deps = ["//tensorflow:tensorflow_py"],
)

py_test(
    name = "deprecation_test",
    srcs = ["python/framework/deprecation_test.py"],
    srcs_version = "PY2AND3",
    deps = ["//tensorflow:tensorflow_py"],
)

py_test(
    name = "tensor_util_test",
    srcs = ["python/framework/tensor_util_test.py"],
    srcs_version = "PY2AND3",
    deps = ["//tensorflow:tensorflow_py"],
)

py_test(
    name = "variables_test",
    size = "small",
    srcs = ["python/ops/variables_test.py"],
    srcs_version = "PY2AND3",
    deps = ["//tensorflow:tensorflow_py"],
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

# Description:
#   contains parts of TensorFlow that are experimental or unstable and which are not supported.

package(default_visibility = [
    "//learning/brain:__subpackages__",
    "//tensorflow:__subpackages__",
    "//tensorflow_model_optimization:__subpackages__",
])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "py_test")
load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")
load("//tensorflow:tensorflow.bzl", "tf_gen_op_wrapper_py")
load("//tensorflow:tensorflow.bzl", "tf_gen_op_libs")
load("//tensorflow:tensorflow.bzl", "tf_kernel_library")
load("//tensorflow:tensorflow.bzl", "cuda_py_test")
load("//tensorflow:tensorflow.bzl", "tf_custom_op_py_library")

tf_custom_op_py_library(
    name = "framework_py",
    srcs = [
        "__init__.py",
        "python/framework/__init__.py",
        "python/framework/checkpoint_utils.py",
        "python/framework/experimental.py",
        "python/framework/graph_util.py",
        "python/framework/tensor_util.py",
        "python/ops/__init__.py",
        "python/ops/arg_scope.py",
        "python/ops/audio_ops.py",
        "python/ops/checkpoint_ops.py",
        "python/ops/critical_section_ops.py",
        "python/ops/ops.py",
        "python/ops/prettyprint_ops.py",
        "python/ops/script_ops.py",
        "python/ops/sort_ops.py",
        "python/ops/variables.py",
    ],
    dso = [
        ":python/ops/_variable_ops.so",
    ],
    kernels = [
        ":variable_kernels",
        ":variable_ops_op_lib",
    ],
    srcs_version = "PY2AND3",
    visibility = [
        "//learning/brain:__subpackages__",
        "//tensorflow:__subpackages__",
        "//tensorflow_estimator:__subpackages__",
        "//video/youtube/personalization:__subpackages__",
    ],
    deps = [
        ":gen_variable_ops",
        "//tensorflow/contrib/util:util_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:audio_ops_gen",
        "//tensorflow/python:check_ops",
        "//tensorflow/python:checkpoint_ops_gen",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:io_ops",
        "//tensorflow/python:logging_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:pywrap_tensorflow",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:script_ops",
        "//tensorflow/python:smart_cond",
        "//tensorflow/python:sort_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:state_ops_gen",
        "//tensorflow/python:tensor_array_ops",
        "//tensorflow/python:tensor_util",
        "//tensorflow/python:training",
        "//tensorflow/python:util",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:function",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

tf_kernel_library(
    name = "variable_kernels",
    srcs = [
        "kernels/zero_initializer_op.cc",
        "kernels/zero_initializer_op.h",
    ],
    gpu_srcs = [
        "kernels/zero_initializer_op_gpu.cu.cc",
        "kernels/zero_initializer_op.h",
    ],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_headers_lib",
        "//third_party/eigen3",
    ],
    alwayslink = 1,
)

tf_custom_op_library(
    name = "python/ops/_variable_ops.so",
    srcs = [
        "kernels/zero_initializer_op.cc",
        "kernels/zero_initializer_op.h",
        "ops/variable_ops.cc",
    ],
    gpu_srcs = [
        "kernels/zero_initializer_op_gpu.cu.cc",
        "kernels/zero_initializer_op.h",
    ],
)

tf_gen_op_libs(
    op_lib_names = [
        "variable_ops",
    ],
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
    deps = [
        ":framework_py",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "checkpoint_utils_test",
    size = "small",
    srcs = ["python/framework/checkpoint_utils_test.py"],
    srcs_version = "PY2AND3",
    tags = ["manual"],  # http://b/30468735
    deps = [
        ":framework_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:partitioned_variables",
        "//tensorflow/python:training",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
    ],
)

cuda_py_test(
    name = "critical_section_test",
    size = "medium",
    srcs = ["python/ops/critical_section_test.py"],
    additional_deps = [
        "//tensorflow/python:client_testlib",
        ":framework_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:gradients",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:tensor_array_ops",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/eager:context",
    ],
)

py_test(
    name = "ops_test",
    size = "small",
    srcs = ["python/ops/ops_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":framework_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
    ],
)

py_test(
    name = "prettyprint_ops_test",
    size = "small",
    srcs = ["python/ops/prettyprint_ops_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":framework_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:sparse_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:tensor_array_ops",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "experimental_test",
    srcs = ["python/framework/experimental_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":framework_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:platform",
    ],
)

py_test(
    name = "graph_util_test",
    srcs = ["python/framework/graph_util_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":framework_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:platform",
    ],
)

py_test(
    name = "tensor_util_test",
    srcs = ["python/framework/tensor_util_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":framework_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "variables_test",
    size = "small",
    srcs = ["python/ops/variables_test.py"],
    srcs_version = "PY2AND3",
    tags = ["manual"],
    deps = [
        ":framework_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:partitioned_variables",
        "//tensorflow/python:platform",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:session",
        "//tensorflow/python:training",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
    ],
)

filegroup(
    name = "checkpoint_ops_testdata",
    srcs = [
        "testdata/bundle_checkpoint.data-00000-of-00001",
        "testdata/bundle_checkpoint.index",
        "testdata/bundle_checkpoint_vocab.txt",
        "testdata/bundle_checkpoint_vocab_with_oov.txt",
        "testdata/keyword.txt",
        "testdata/keyword_new.txt",
    ],
)

py_test(
    name = "checkpoint_ops_test",
    size = "medium",
    srcs = ["python/ops/checkpoint_ops_test.py"],
    data = [":checkpoint_ops_testdata"],
    srcs_version = "PY2AND3",
    tags = ["no_pip"],
    deps = [
        ":framework_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:partitioned_variables",
        "//tensorflow/python:platform",
        "//tensorflow/python:training",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
    ],
)

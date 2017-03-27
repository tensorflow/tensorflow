# Description:
#   contains parts of TensorFlow that are experimental or unstable and which are not supported.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = [
    "//learning/brain:__subpackages__",
    "//tensorflow:__subpackages__",
])

load("//tensorflow:tensorflow.bzl", "cuda_py_test")
load("//tensorflow:tensorflow.bzl", "tf_custom_op_py_library")
load("//tensorflow:tensorflow.bzl", "py_test")
load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")
load("//tensorflow:tensorflow.bzl", "tf_gen_op_libs")
load("//tensorflow:tensorflow.bzl", "tf_gen_op_wrapper_py")
load("//tensorflow:tensorflow.bzl", "tf_kernel_library")

tf_custom_op_library(
    # TODO(sibyl-Mooth6ku,ptucker): Understand why 'python/ops/_' is needed and fix it.
    name = "python/ops/_bucketization_op.so",
    srcs = [
        "ops/bucketization_op.cc",
    ],
    deps = [
        "//tensorflow/contrib/layers/kernels:bucketization_kernel",
    ],
)

tf_gen_op_libs(
    op_lib_names = ["bucketization_op"],
)

tf_gen_op_wrapper_py(
    name = "bucketization_op",
    deps = [":bucketization_op_op_lib"],
)

tf_kernel_library(
    name = "bucketization_op_kernel",
    deps = [
        "//tensorflow/contrib/layers/kernels:bucketization_kernel",
        "//tensorflow/core:framework",
    ],
    alwayslink = 1,
)

tf_custom_op_library(
    # TODO(sibyl-Mooth6ku,ptucker): Understand why 'python/ops/_' is needed and fix it.
    name = "python/ops/_sparse_feature_cross_op.so",
    srcs = [
        "ops/sparse_feature_cross_op.cc",
    ],
    deps = [
        "//tensorflow/contrib/layers/kernels:sparse_feature_cross_kernel",
    ],
)

tf_gen_op_libs(
    op_lib_names = ["sparse_feature_cross_op"],
)

tf_gen_op_wrapper_py(
    name = "sparse_feature_cross_op",
    deps = [":sparse_feature_cross_op_op_lib"],
)

tf_kernel_library(
    name = "sparse_feature_cross_op_kernel",
    deps = [
        "//tensorflow/contrib/layers/kernels:sparse_feature_cross_kernel",
        "//tensorflow/core:framework",
    ],
    alwayslink = 1,
)

tf_custom_op_py_library(
    name = "layers_py",
    srcs = [
        "__init__.py",
        "python/layers/__init__.py",
        "python/layers/embedding_ops.py",
        "python/layers/encoders.py",
        "python/layers/feature_column.py",
        "python/layers/feature_column_ops.py",
        "python/layers/initializers.py",
        "python/layers/layers.py",
        "python/layers/optimizers.py",
        "python/layers/regularizers.py",
        "python/layers/summaries.py",
        "python/layers/target_column.py",
        "python/layers/utils.py",
        "python/ops/bucketization_op.py",
        "python/ops/sparse_feature_cross_op.py",
        "python/ops/sparse_ops.py",
    ],
    dso = [
        ":python/ops/_bucketization_op.so",
        ":python/ops/_sparse_feature_cross_op.so",
    ],
    kernels = [
        ":bucketization_op_kernel",
        ":sparse_feature_cross_op_kernel",
        ":bucketization_op_op_lib",
        ":sparse_feature_cross_op_op_lib",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":bucketization_op",
        ":sparse_feature_cross_op",
        "//tensorflow/contrib/framework:framework_py",
        "//tensorflow/contrib/lookup:lookup_py",
        "//tensorflow/contrib/losses:losses_py",
        "//tensorflow/contrib/metrics:metrics_py",
        "//tensorflow/contrib/util:util_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:check_ops",
        "//tensorflow/python:clip_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:embedding_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:layers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:parsing_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:sparse_ops",
        "//tensorflow/python:standard_ops",
        "//tensorflow/python:string_ops",
        "//tensorflow/python:summary",
        "//tensorflow/python:training",
        "//tensorflow/python:util",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "@six_archive//:six",
    ],
)

cuda_py_test(
    name = "layers_test",
    size = "medium",
    srcs = ["python/layers/layers_test.py"],
    additional_deps = [
        ":layers_py",
        "//third_party/py/numpy",
        "//tensorflow/contrib/framework:framework_py",
        "//tensorflow/contrib/losses:losses_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:gradients",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:partitioned_variables",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:template",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
    ],
)

py_test(
    name = "regularizers_test",
    size = "small",
    srcs = ["python/layers/regularizers_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":layers_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "initializers_test",
    size = "small",
    srcs = ["python/layers/initializers_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":layers_py",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "optimizers_test",
    srcs = ["python/layers/optimizers_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":layers_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:training",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "summaries_test",
    size = "small",
    srcs = ["python/layers/summaries_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":layers_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "feature_column_test",
    size = "small",
    srcs = ["python/layers/feature_column_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":layers_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:parsing_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:training",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "feature_column_ops_test",
    size = "medium",
    srcs = ["python/layers/feature_column_ops_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":layers_py",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:data_flow_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:gradients",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:partitioned_variables",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "target_column_test",
    size = "small",
    srcs = ["python/layers/target_column_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":layers_py",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "bucketization_op_test",
    size = "small",
    srcs = ["python/kernel_tests/bucketization_op_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":layers_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "sparse_feature_cross_op_test",
    size = "medium",
    srcs = ["python/kernel_tests/sparse_feature_cross_op_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":layers_py",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:sparse_ops",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "embedding_ops_test",
    size = "small",
    srcs = ["python/layers/embedding_ops_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":layers_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:partitioned_variables",
        "//tensorflow/python:platform_test",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "utils_test",
    size = "small",
    srcs = ["python/layers/utils_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":layers_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "sparse_ops_test",
    size = "small",
    srcs = ["python/ops/sparse_ops_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":layers_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "encoders_test",
    size = "small",
    srcs = ["python/layers/encoders_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":layers_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
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

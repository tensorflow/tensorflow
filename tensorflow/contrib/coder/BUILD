# Description:
#   Contains ops related to data compression.

package(default_visibility = [
    "//learning/brain:__subpackages__",
    "//research/vision/piedpiper:__subpackages__",
    "//tensorflow:__subpackages__",
])

licenses(["notice"])  # Apache 2.0

load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
    "tf_custom_op_library",
    "tf_custom_op_py_library",
    "tf_gen_op_libs",
    "tf_gen_op_wrapper_py",
    "tf_kernel_library",
    "tf_py_test",
)

cc_library(
    name = "range_coder",
    srcs = [
        "kernels/range_coder.cc",
    ],
    hdrs = [
        "kernels/range_coder.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "range_coder_test",
    size = "small",
    srcs = ["kernels/range_coder_test.cc"],
    deps = [
        ":range_coder",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_gen_op_libs(
    op_lib_names = ["coder_ops"],
    deps = [
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "range_coder_ops_util",
    srcs = ["kernels/range_coder_ops_util.cc"],
    hdrs = ["kernels/range_coder_ops_util.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
)

tf_kernel_library(
    name = "range_coder_ops",
    srcs = [
        "kernels/range_coder_ops.cc",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":coder_ops_op_lib",
        ":range_coder",
        ":range_coder_ops_util",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
    alwayslink = 1,
)

tf_cc_test(
    name = "range_coder_ops_test",
    size = "small",
    srcs = ["kernels/range_coder_ops_test.cc"],
    deps = [
        ":range_coder",
        ":range_coder_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels:ops_testutil",
    ],
)

tf_kernel_library(
    name = "pmf_to_cdf_op",
    srcs = ["kernels/pmf_to_cdf_op.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":coder_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "pmf_to_cdf_op_test",
    size = "small",
    srcs = ["kernels/pmf_to_cdf_op_test.cc"],
    deps = [
        ":pmf_to_cdf_op",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels:ops_testutil",
    ],
)

cc_library(
    name = "all_ops",
    deps = [":coder_ops_op_lib"],
)

cc_library(
    name = "all_kernels",
    deps = [
        ":pmf_to_cdf_op",
        ":range_coder_ops",
    ],
)

tf_custom_op_library(
    name = "python/ops/_coder_ops.so",
    srcs = [
        "kernels/pmf_to_cdf_op.cc",
        "kernels/range_coder.cc",
        "kernels/range_coder.h",
        "kernels/range_coder_ops.cc",
        "kernels/range_coder_ops_util.cc",
        "kernels/range_coder_ops_util.h",
        "ops/coder_ops.cc",
    ],
)

tf_gen_op_wrapper_py(
    name = "gen_coder_ops",
    out = "python/ops/gen_coder_ops.py",
    deps = [":coder_ops_op_lib"],
)

py_library(
    name = "coder_py",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":coder_ops_py",
    ],
)

tf_custom_op_py_library(
    name = "coder_ops_py",
    srcs = [
        "python/ops/coder_ops.py",
    ],
    dso = [
        ":python/ops/_coder_ops.so",
    ],
    kernels = [
        ":all_kernels",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":gen_coder_ops",
        "//tensorflow/contrib/util:util_py",
    ],
)

tf_py_test(
    name = "coder_ops_py_test",
    srcs = [
        "python/ops/coder_ops_test.py",
    ],
    additional_deps = [
        ":coder_ops_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_ops",
    ],
    main = "python/ops/coder_ops_test.py",
)

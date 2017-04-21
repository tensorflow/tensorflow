# Description:
# An example for custom op and kernel defined as a TensorFlow plugin.

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "tf_gen_op_libs")
load("//tensorflow:tensorflow.bzl", "tf_py_test")
load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")
load("//tensorflow:tensorflow.bzl", "tf_gen_op_wrapper_py")

filegroup(
    name = "custom_op_sources",
    srcs = glob(
        [
            "core/ops/*.cc",
            "core/kernels/*.cc",
        ],
        exclude = [
            "core/ops/*_test.cc",
            "core/kernels/*_test.cc",
        ],
    ),
)

filegroup(
    name = "custom_op_headers",
    srcs = glob(
        [
            "core/kernels/*.h",
            "core/ops/*.h",
        ],
    ),
)

cc_library(
    name = "all_ops",
    srcs = [":custom_op_sources"],
    hdrs = [":custom_op_headers"],
    deps = [
        "//tensorflow/core:framework_headers_lib",
    ],
    alwayslink = 1,
)

tf_custom_op_library(
    name = "python/ops/_shuffle_op.so",
    srcs = [":custom_op_sources"],
)

py_library(
    name = "init_py",
    srcs = [
        "__init__.py",
        "python/__init__.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":shuffle_op_py",
    ],
)

#tf_gen_op_libs(
#    op_lib_names = ["shuffle_op"],
#)

#tf_gen_op_wrapper_py(
#    name = "gen_shuffle_op",
#    out = "python/ops/gen_shuffle_op.py",
#    deps = [":shuffle_op_op_lib"],
#)

py_library(
    name = "shuffle_op_py",
    srcs = ["python/ops/shuffle_op.py"],
    data = ["python/ops/_shuffle_op.so"],
    srcs_version = "PY2AND3",
#    deps = [
#        ":gen_shuffle_op",
#    ],
)

# tf_gen_op_wrapper_py(
    # name = "gen_shuffle_op",
    # out = "python/ops/gen_shuffle_op.py",
    # deps = [":python/ops/_shuffle_op.so"],
# )

# tf_py_test(
    # name = "shuffle_test",
    # size = "small",
    # srcs = ["shuffle_test.py"],
    # additional_deps = ["//tensorflow:tensorflow_py"],
    # data = [":_shuffle_op.so"],
# )

#py_library(
    #name = "shuffle_py",
    #srcs = [
        #"__init__.py",
        # "python/ops/gen_shuffle_op.py",
    #],
    #data = [
        #":python/ops/_shuffle_op.so",
    #],
    #srcs_version = "PY2AND3",
    #visibility = ["//visibility:public"],
#)

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

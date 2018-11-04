package(default_visibility = ["//tensorflow:__subpackages__"])

load(
    "//tensorflow:tensorflow.bzl",
    "tf_custom_op_library",
    "tf_gen_op_libs",
    "tf_gen_op_wrapper_py",
    "tf_custom_op_py_library",
)

load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")


tf_custom_op_library(
    name = "_lib_ops.so",
    srcs = [
        "ops/image_pipe_ops.cc",
    ],
    deps = ["//tensorflow/core:jpeg_internal"] + if_cuda([
        "//tensorflow/core:gpu_headers_lib",
    ]),
)

tf_gen_op_libs(
    op_lib_names = ["image_pipe_ops"],
    deps = [
        "//tensorflow/core:stream_executor",
        "//tensorflow/core:jpeg_internal",
        "//tensorflow/core:lib",
    ],
)

tf_gen_op_wrapper_py(
    name = "image_pipe_ops",
    deps = [":image_pipe_ops_op_lib"],
)

tf_custom_op_py_library(
    name = "setup_py",
    srcs = [
        "__init__.py",
    ],
    kernels = [],
    deps = [
        ":image_pipe_ops",
        "//tensorflow/contrib/util:util_py",
        "//tensorflow/python:device",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:util",
        "//tensorflow/python/eager:context",
    ],
    dso = [":_lib_ops.so"],
    visibility = ["//visibility:public"],
)

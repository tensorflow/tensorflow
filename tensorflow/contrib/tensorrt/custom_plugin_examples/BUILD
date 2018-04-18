package(default_visibility = ["//tensorflow:__subpackages__"])

load(
    "//tensorflow:tensorflow.bzl",
    "tf_custom_op_library",
    "tf_cuda_library",
    "tf_gen_op_libs",
    "tf_gen_op_wrapper_py",
    "tf_py_wrap_cc",
    "tf_copts",
)
load(
    "@local_config_tensorrt//:build_defs.bzl",
    "if_tensorrt",
)
load("//tensorflow:tensorflow.bzl", "tf_custom_op_py_library")
load("//tensorflow:tensorflow.bzl", "tf_kernel_library")

tf_kernel_library(
    name = "_inc_op_plugin_kernel",
    srcs = [
        "inc_op_plugin.cc",
    ],
    hdrs = [
    ],
    gpu_srcs = [
        "inc_op_kernel.cu.cc",
        "inc_op_kernel.h",
        "inc_op_plugin.h",
    ],
    deps = if_tensorrt([
        "@local_config_tensorrt//:nv_infer",
        "//tensorflow/contrib/tensorrt:trt_plugins",
    ]),
)

tf_gen_op_libs(
    op_lib_names = [
        "inc_op",
    ],
    deps = if_tensorrt([
        "@local_config_tensorrt//:nv_infer",
        "//tensorflow/contrib/tensorrt:trt_plugins",
    ]),
)

tf_gen_op_wrapper_py(
    name = "inc_op",
    gen_locally = True,
    deps = [
        ":inc_op_op_lib",
    ],
)

tf_py_wrap_cc(
    name = "plugin_wrap",
    srcs = [
        "plugin_wrap.i",
    ],
    copts = tf_copts(),
    deps = [
        ":_inc_op_plugin_kernel",
        "//tensorflow/core:framework_lite",
        "//util/python:python_headers",
    ],
)

tf_custom_op_library(
    name = "_inc_op.so",
    srcs = ["ops/inc_op.cc"],
    deps = [
        "//tensorflow/core:lib_proto_parsing",
    ] + if_tensorrt([
        "//tensorflow/contrib/tensorrt:trt_plugins",
    ]),
)

tf_custom_op_py_library(
    name = "inc_op_loader",
    srcs = ["inc_op.py"],
    dso = [
        ":_inc_op.so",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:resources",
    ],
)

py_library(
    name = "inc_op_py",
    srcs_version = "PY2AND3",
    deps = [
        ":inc_op",
        ":inc_op_loader",
    ],
)

py_library(
    name = "init_py",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":inc_op_py",
        ":plugin_wrap",
    ],
)

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "tf_py_wrap_cc")

cc_library(
    name = "interpreter_wrapper_lib",
    srcs = ["interpreter_wrapper.cc"],
    hdrs = ["interpreter_wrapper.h"],
    deps = [
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite/kernels:builtin_ops",
        "//tensorflow/core:lib",
        "//third_party/py/numpy:headers",
        "//third_party/python_runtime:headers",
        "@com_google_absl//absl/memory",
    ],
)

tf_py_wrap_cc(
    name = "tensorflow_wrap_interpreter_wrapper",
    srcs = [
        "interpreter_wrapper.i",
    ],
    deps = [
        ":interpreter_wrapper_lib",
        "//third_party/python_runtime:headers",
    ],
)

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
        ":python_error_reporter",
        ":python_utils",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:builtin_ops",
        "//third_party/py/numpy:headers",
        "//third_party/python_runtime:headers",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "python_error_reporter",
    srcs = ["python_error_reporter.cc"],
    hdrs = ["python_error_reporter.h"],
    deps = [
        "//tensorflow/lite/core/api",
        "//third_party/python_runtime:headers",
    ],
)

cc_library(
    name = "python_utils",
    srcs = ["python_utils.cc"],
    hdrs = ["python_utils.h"],
    deps = [
        "//tensorflow/lite:framework",
        "//third_party/py/numpy:headers",
        "//third_party/python_runtime:headers",
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

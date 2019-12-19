load("//tensorflow:tensorflow.bzl", "tf_py_wrap_cc")
load("//tensorflow/lite:build_def.bzl", "if_tflite_experimental_runtime", "tflite_experimental_runtime_linkopts")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "numpy",
    srcs = ["numpy.cc"],
    hdrs = ["numpy.h"],
    deps = [
        "//tensorflow/lite:string_util",
        "//tensorflow/lite/c:common",
        "//third_party/py/numpy:headers",
        "//third_party/python_runtime:headers",
    ],
)

cc_library(
    name = "interpreter_wrapper_lib",
    srcs = ["interpreter_wrapper.cc"],
    hdrs = [
        "interpreter_wrapper.h",
        "//tensorflow/lite/experimental/tflite_api_dispatcher:tflite_api_dispatcher.h",
    ],
    defines = if_tflite_experimental_runtime(
        if_false = [],
        if_true = ["TFLITE_EXPERIMENTAL_RUNTIME"],
    ),
    deps = [
        ":numpy",
        ":python_error_reporter",
        ":python_utils",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings:str_format",
        "//third_party/python_runtime:headers",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:string_util",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/kernels:builtin_ops",
    ] + tflite_experimental_runtime_linkopts(),
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
        "//tensorflow/lite:string_util",
        "//third_party/python_runtime:headers",
    ],
)

tf_py_wrap_cc(
    name = "tensorflow_wrap_interpreter_wrapper",
    srcs = [
        "interpreter_wrapper.i",
    ],
    copts = ["-fexceptions"],
    deps = [
        ":interpreter_wrapper_lib",
        "//third_party/python_runtime:headers",
    ],
)

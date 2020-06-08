load("//tensorflow:tensorflow.bzl", "pybind_extension")

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
        "//third_party/python_runtime:headers",  # buildcleaner: keep
    ],
)

cc_library(
    name = "interpreter_wrapper_lib",
    srcs = ["interpreter_wrapper.cc"],
    hdrs = [
        "interpreter_wrapper.h",
    ],
    deps = [
        ":numpy",
        ":python_error_reporter",
        ":python_utils",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:string_util",
        "//tensorflow/lite:util",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/experimental/tflite_api_dispatcher",
        "//tensorflow/lite/kernels:builtin_ops",
        "//third_party/python_runtime:headers",  # buildcleaner: keep
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "python_error_reporter",
    srcs = ["python_error_reporter.cc"],
    hdrs = ["python_error_reporter.h"],
    deps = [
        "//tensorflow/lite/core/api",
        "//third_party/python_runtime:headers",  # buildcleaner: keep
    ],
)

cc_library(
    name = "python_utils",
    srcs = ["python_utils.cc"],
    hdrs = ["python_utils.h"],
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite:string_util",
        "//third_party/python_runtime:headers",  # buildcleaner: keep
    ],
)

config_setting(
    name = "tflite_pip_with_flex",
    define_values = {
        "tflite_pip_with_flex": "true",
    },
)

pybind_extension(
    name = "_pywrap_tensorflow_interpreter_wrapper",
    srcs = [
        "interpreter_wrapper_pybind11.cc",
    ],
    hdrs = ["interpreter_wrapper.h"],
    link_in_framework = True,
    module_name = "_pywrap_tensorflow_interpreter_wrapper",
    deps = [
        ":interpreter_wrapper_lib",
        "@pybind11",
        "//third_party/python_runtime:headers",
        "//tensorflow/lite:framework_lib",
        "//tensorflow/lite/experimental/tflite_api_dispatcher",
        "//tensorflow/python:pybind11_lib",
    ] + select({
        ":tflite_pip_with_flex": ["//tensorflow/lite/delegates/flex:delegate"],
        "//conditions:default": [],
    }),
)

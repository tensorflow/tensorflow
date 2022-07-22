load("//tensorflow:tensorflow.bzl", "pybind_extension")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "get_compatible_with_portable")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "numpy",
    srcs = ["numpy.cc"],
    hdrs = ["numpy.h"],
    compatible_with = get_compatible_with_portable(),
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
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":numpy",
        ":python_error_reporter",
        ":python_utils",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:shared_library",
        "//tensorflow/lite:string_util",
        "//tensorflow/lite:util",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/kernels:reference_ops",
        "//tensorflow/lite/kernels/internal:compatibility",
        "//third_party/python_runtime:headers",  # buildcleaner: keep
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "python_error_reporter",
    srcs = ["python_error_reporter.cc"],
    hdrs = ["python_error_reporter.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tensorflow/lite:stateful_error_reporter",
        "//third_party/python_runtime:headers",  # buildcleaner: keep
    ],
)

cc_library(
    name = "python_utils",
    srcs = ["python_utils.cc"],
    hdrs = ["python_utils.h"],
    compatible_with = get_compatible_with_portable(),
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
    compatible_with = get_compatible_with_portable(),
    link_in_framework = True,
    deps = [
        ":interpreter_wrapper_lib",
        "@pybind11",
        "//third_party/python_runtime:headers",
        "//tensorflow/lite:framework_lib",
        "//tensorflow/python:pybind11_lib",
    ] + select({
        ":tflite_pip_with_flex": ["//tensorflow/lite/delegates/flex:delegate"],
        "//conditions:default": [],
    }),
)

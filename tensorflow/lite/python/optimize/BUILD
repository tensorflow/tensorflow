load("//tensorflow:tensorflow.bzl", "pybind_extension")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "calibration_wrapper_lib",
    srcs = ["calibration_wrapper.cc"],
    hdrs = ["calibration_wrapper.h"],
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/python/interpreter_wrapper:numpy",
        "//tensorflow/lite/python/interpreter_wrapper:python_error_reporter",
        "//tensorflow/lite/python/interpreter_wrapper:python_utils",
        "//tensorflow/lite/tools/optimize:quantization_wrapper_utils",
        "//tensorflow/lite/tools/optimize:quantize_model",
        "//tensorflow/lite/tools/optimize/calibration:calibration_reader",
        "//tensorflow/lite/tools/optimize/calibration:calibrator_lib",
        "//third_party/python_runtime:headers",  # buildcleaner: keep
        "@com_google_absl//absl/memory",
    ],
)

pybind_extension(
    name = "_pywrap_tensorflow_lite_calibration_wrapper",
    srcs = [
        "calibration_wrapper_pybind11.cc",
    ],
    hdrs = ["calibration_wrapper.h"],
    link_in_framework = True,
    module_name = "_pywrap_tensorflow_lite_calibration_wrapper",
    deps = [
        ":calibration_wrapper_lib",
        "//tensorflow/lite:framework_lib",
        "//tensorflow/python:pybind11_lib",
        "//third_party/python_runtime:headers",
        "@pybind11",
    ],
)

py_library(
    name = "calibrator",
    srcs = [
        "calibrator.py",
    ],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":_pywrap_tensorflow_lite_calibration_wrapper",  # buildcleaner: keep
        "//tensorflow/python:dtypes",
        "//tensorflow/python:util",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "calibrator_test",
    srcs = ["calibrator_test.py"],
    data = [
        ":test_data",
        "//tensorflow/lite:testdata/multi_add.bin",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = ["no_oss"],
    deps = [
        ":calibrator",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
        "@six_archive//:six",
    ],
)

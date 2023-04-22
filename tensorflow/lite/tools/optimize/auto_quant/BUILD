# Auto Selective Quantization modules.
load("//tensorflow:tensorflow.bzl", "pytype_strict_library")
load("//tensorflow:tensorflow.bzl", "py_strict_test")

package(
    licenses = ["notice"],
)

pytype_strict_library(
    name = "auto_quant",
    srcs = ["auto_quant.py"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/lite/tools/optimize/debugging/python:debugger",
        "//tensorflow/python/saved_model:loader",
        "//third_party/py/numpy",
    ],
)

py_strict_test(
    name = "auto_quant_test",
    srcs = ["auto_quant_test.py"],
    deps = [
        ":auto_quant",
        "//tensorflow:tensorflow_py",
        "//tensorflow/lite/python:schema_py",
        "//tensorflow/lite/tools:flatbuffer_utils",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

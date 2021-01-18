load("//tensorflow:tensorflow.bzl", "pybind_extension")

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],  # Apache 2.0
)

pybind_extension(
    name = "_pywrap_tensorflow_lite_sanitizers",
    srcs = [
        "sanitizers_pybind11.cc",
    ],
    link_in_framework = True,
    module_name = "_pywrap_tensorflow_lite_sanitizers",
    deps = [
        "//tensorflow/python:pybind11_lib",
        "@pybind11",
    ],
)

load("//tensorflow:tensorflow.bzl", "pybind_extension")

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],  # Apache 2.0
)

pybind_extension(
    name = "_pywrap_flatbuffers",
    srcs = [
        "flatbuffers_lib.cc",
    ],
    features = ["-use_header_modules"],
    module_name = "_pywrap_flatbuffers",
    deps = [
        "//tensorflow/python:pybind11_lib",
        "//third_party/python_runtime:headers",
        "@flatbuffers",
        "@pybind11",
    ],
)

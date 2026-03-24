load("@xla//third_party/rules_python/python:py_library.bzl", "py_library")
load("//tensorflow:tensorflow.bzl", "py_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:LICENSE"],
    licenses = ["notice"],
)

py_library(
    name = "authoring",
    srcs = [
        "authoring.py",
    ],
    strict_deps = True,
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/mlir/lite/metrics:converter_error_data_proto_py",
        "//tensorflow/lite/python:convert",
        "//tensorflow/lite/python:lite",
        "//tensorflow/python/util:tf_export",
    ],
)

py_test(
    name = "authoring_test",
    srcs = ["authoring_test.py"],
    strict_deps = True,
    deps = [
        ":authoring",
        #internal proto upb dep
        "//tensorflow:tensorflow_py",
        "//tensorflow/lite/python:lite",
    ],
)

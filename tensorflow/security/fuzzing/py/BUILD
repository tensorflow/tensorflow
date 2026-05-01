load("@xla//third_party/rules_python/python:py_library.bzl", "py_library")

# copybara:uncomment package(default_applicable_licenses = ["//tensorflow:license"])

# A library of python types used to annotate ops
py_library(
    name = "annotation_types",
    srcs = ["annotation_types.py"],
    strict_deps = True,
    visibility = ["//tensorflow/python/framework:__pkg__"],
    deps = ["//tensorflow/python/framework:dtypes"],
)

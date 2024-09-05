load("//tensorflow:strict.default.bzl", "py_strict_library")

# copybara:uncomment package(default_applicable_licenses = ["//tensorflow:license"])

# A library of python types used to annotate ops
py_strict_library(
    name = "annotation_types",
    srcs = ["annotation_types.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow/python/framework:__pkg__"],
    deps = ["//tensorflow/python/framework:dtypes"],
)

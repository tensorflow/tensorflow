load("@xla//third_party/rules_python/python:py_library.bzl", "py_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

py_library(
    name = "data",
    srcs = ["__init__.py"],
    strict_deps = True,
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/python/data/experimental",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:iterator_ops",
        "//tensorflow/python/data/ops:options",
        "//tensorflow/python/data/ops:readers",
    ],
)

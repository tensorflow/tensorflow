load("@xla//third_party/rules_python/python:py_library.bzl", "py_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

py_library(
    name = "experimental",
    srcs = [
        "__init__.py",
    ],
    strict_deps = True,
    deps = [
        "//tensorflow/python/tpu:tpu_strategy_util",
    ],
)

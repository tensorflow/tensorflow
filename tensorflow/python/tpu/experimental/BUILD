load("//tensorflow:strict.default.bzl", "py_strict_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

py_strict_library(
    name = "experimental",
    srcs = [
        "__init__.py",
    ],
    deps = [
        "//tensorflow/python/tpu:tpu_strategy_util",
    ],
)

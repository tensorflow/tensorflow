# python/lib package

load("@xla//third_party/rules_python/python:py_library.bzl", "py_library")

visibility = [
    "//tensorflow:__subpackages__",
]

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = visibility,
    licenses = ["notice"],
)

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
    strict_deps = True,
    visibility = ["//tensorflow:__subpackages__"],
    deps = [],
)

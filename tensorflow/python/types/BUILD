load("//tensorflow:pytype.default.bzl", "pytype_strict_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

# Important: this is a leaf library. It may not have any new dependencies inside TF proper.
# The sole exception is tf_export, to allow exporting symbols into the public namespace.
pytype_strict_library(
    name = "core",
    srcs = [
        "__init__.py",
        "core.py",
    ],
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        ":doc_typealias",
        "//tensorflow/python:pywrap_tensorflow",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
        "@pypi_typing_extensions//:pkg",
    ],
)

pytype_strict_library(
    name = "data",
    srcs = ["data.py"],
    visibility = [
        "//tensorflow:__subpackages__",
    ],
    deps = ["//tensorflow/python/util:tf_export"],
)

pytype_strict_library(
    name = "distribute",
    srcs = ["distribute.py"],
    visibility = [
        "//tensorflow:__subpackages__",
    ],
    deps = [
        "//tensorflow/python/util:tf_export",
        "//tensorflow/tools/docs:doc_controls",
    ],
)

pytype_strict_library(
    name = "internal",
    srcs = ["internal.py"],
    visibility = [
        "//tensorflow:__subpackages__",
    ],
)

pytype_strict_library(
    name = "trace",
    srcs = ["trace.py"],
    visibility = [
        "//tensorflow:__subpackages__",
    ],
    deps = [
        ":core",
        "//tensorflow/python/util:tf_export",
        "//tensorflow/tools/docs:doc_controls",
        "@pypi_typing_extensions//:pkg",
    ],
)

pytype_strict_library(
    name = "doc_typealias",
    srcs = [
        "doc_typealias.py",
    ],
    visibility = [
        "//tensorflow:__subpackages__",
    ],
    deps = [],
)

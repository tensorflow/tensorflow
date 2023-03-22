load("//tensorflow:strict.default.bzl", "py_strict_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//visibility:private"],
)

py_strict_library(
    name = "autograph",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/python:util",
        "//tensorflow/python/autograph/converters:__init__",
        "//tensorflow/python/autograph/core:converter",
        "//tensorflow/python/autograph/impl:api",
        "//tensorflow/python/autograph/lang:directives",
        "//tensorflow/python/autograph/lang:special_functions",
        "//tensorflow/python/autograph/operators",
        "//tensorflow/python/autograph/utils:__init__",
        "//tensorflow/python/autograph/utils:ag_logging",
    ],
)

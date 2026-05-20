load("@xla//third_party/rules_python/python:py_library.bzl", "py_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

py_library(
    name = "autograph",
    srcs = [
        "__init__.py",
    ],
    strict_deps = True,
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/python/autograph/core:converter",
        "//tensorflow/python/autograph/impl:api",
        "//tensorflow/python/autograph/lang:directives",
        "//tensorflow/python/autograph/lang:special_functions",
        "//tensorflow/python/autograph/operators:__init__",
        "//tensorflow/python/autograph/pyct/common_transformers",
        "//tensorflow/python/autograph/utils:__init__",
        "//tensorflow/python/autograph/utils:ag_logging",
        "//tensorflow/python/util:all_util",
    ],
)

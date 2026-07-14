load("@xla//third_party/rules_python/python:py_library.bzl", "py_library")
load("//tensorflow:tensorflow.bzl", "py_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

py_library(
    name = "codegen",
    srcs = [
        "codegen.py",
    ],
    strict_deps = True,
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/python/autograph/pyct:templates",
        "//third_party/py/numpy",
        "@pypi//gast",
    ],
)

py_library(
    name = "decorators",
    srcs = ["decorators.py"],
    strict_deps = True,
    visibility = ["//visibility:public"],
)

py_library(
    name = "basic_definitions",
    srcs = ["basic_definitions.py"],
    strict_deps = True,
    visibility = ["//visibility:public"],
)

py_test(
    name = "codegen_test",
    size = "large",
    srcs = ["codegen_test.py"],
    strict_deps = True,
    tags = [
        "manual",
        "no_windows",
        "nomsan",
        "notap",
    ],
    deps = [
        ":codegen",
        #internal proto upb dep
        "//third_party/py/numpy",
        "//tensorflow/python/platform:client_testlib",
    ],
)

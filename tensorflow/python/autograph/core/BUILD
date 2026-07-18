load("@xla//third_party/rules_python/python:py_library.bzl", "py_library")
load("//tensorflow:tensorflow.bzl", "py_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

py_library(
    name = "test_lib",
    srcs = [
        "converter_testing.py",
    ],
    strict_deps = True,
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        ":config",
        ":converter",
        "//tensorflow/python/autograph/impl:api",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_library(
    name = "ag_ctx",
    srcs = ["ag_ctx.py"],
    strict_deps = True,
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        "//tensorflow/python/autograph/utils:ag_logging",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "unsupported_features_checker",
    srcs = ["unsupported_features_checker.py"],
    strict_deps = True,
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        "//tensorflow/python/autograph/pyct:errors",
        "@pypi//gast",
    ],
)

py_library(
    name = "config",
    srcs = ["config.py"],
    strict_deps = True,
    visibility = ["//tensorflow:__subpackages__"],
    deps = [":config_lib"],
)

py_library(
    name = "config_lib",
    srcs = ["config_lib.py"],
    strict_deps = True,
    visibility = ["//visibility:private"],
)

py_library(
    name = "function_wrappers",
    srcs = ["function_wrappers.py"],
    strict_deps = True,
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        ":ag_ctx",
        ":converter",
        "//tensorflow/python/autograph/operators:variables",
        "//tensorflow/python/framework:auto_control_deps",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/util:nest",
    ],
)

py_library(
    name = "converter",
    srcs = ["converter.py"],
    strict_deps = True,
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        "//tensorflow/python/autograph/pyct:anno",
        "//tensorflow/python/autograph/pyct:ast_util",
        "//tensorflow/python/autograph/pyct:parser",
        "//tensorflow/python/autograph/pyct:templates",
        "//tensorflow/python/autograph/pyct:transformer",
        "//tensorflow/python/util:tf_export",
    ],
)

py_test(
    name = "converter_test",
    srcs = ["converter_test.py"],
    strict_deps = True,
    deps = [
        ":converter",
        ":test_lib",
        #internal proto upb dep
        "//tensorflow/python/autograph/pyct:anno",
        "//tensorflow/python/autograph/pyct:loader",
        "//tensorflow/python/autograph/pyct:parser",
        "//tensorflow/python/autograph/pyct:templates",
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_test(
    name = "function_wrappers_test",
    srcs = ["function_wrappers_test.py"],
    strict_deps = True,
    deps = [
        ":converter",
        ":function_wrappers",
        #internal proto upb dep
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
    ],
)

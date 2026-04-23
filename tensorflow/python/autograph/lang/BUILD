load("@xla//third_party/rules_python/python:py_library.bzl", "py_library")
load("//tensorflow:tensorflow.bzl", "py_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

py_library(
    name = "directives",
    srcs = ["directives.py"],
    strict_deps = True,
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "special_functions",
    srcs = ["special_functions.py"],
    strict_deps = True,
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        "//tensorflow/python/autograph/operators:data_structures",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:tensor_util",
    ],
)

py_test(
    name = "special_functions_test",
    srcs = ["special_functions_test.py"],
    strict_deps = True,
    deps = [
        #internal proto upb dep
        "//third_party/py/numpy",
        "//tensorflow/python/autograph/lang:special_functions",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/ops:list_ops",
        "//tensorflow/python/platform:client_testlib",
    ],
)

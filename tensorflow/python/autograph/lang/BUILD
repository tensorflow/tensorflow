load("//tensorflow:strict.default.bzl", "py_strict_library", "py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

py_strict_library(
    name = "directives",
    srcs = ["directives.py"],
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "special_functions",
    srcs = ["special_functions.py"],
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        "//tensorflow/python/autograph/operators:data_structures",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:tensor_util",
    ],
)

py_strict_test(
    name = "special_functions_test",
    srcs = ["special_functions_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
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

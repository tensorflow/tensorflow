licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "py_test")

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)

py_library(
    name = "convert",
    srcs = [
        "builtin_functions.py",
        "call_trees.py",
        "control_flow.py",
        "for_canonicalization.py",
        "logical_expressions.py",
        "print_functions.py",
        "side_effect_guards.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        "@gast_archive//:gast",
    ],
)

py_test(
    name = "call_trees_test",
    srcs = ["call_trees_test.py"],
    deps = [
        ":convert",
        "//tensorflow/contrib/py2tf/pyct",
        "//tensorflow/contrib/py2tf/pyct/static_analysis",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "control_flow_test",
    srcs = ["control_flow_test.py"],
    deps = [
        ":convert",
        "//tensorflow/contrib/py2tf/pyct",
        "//tensorflow/contrib/py2tf/pyct/static_analysis",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "builtin_functions_test",
    srcs = ["builtin_functions_test.py"],
    deps = [
        ":convert",
        "//tensorflow/contrib/py2tf/pyct",
        "//tensorflow/contrib/py2tf/pyct/static_analysis",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "for_canonicalization_test",
    srcs = ["for_canonicalization_test.py"],
    deps = [
        ":convert",
        "//tensorflow/contrib/py2tf/pyct",
        "//tensorflow/contrib/py2tf/pyct/static_analysis",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "logical_expressions_test",
    srcs = ["logical_expressions_test.py"],
    deps = [
        ":convert",
        "//tensorflow/contrib/py2tf/pyct",
        "//tensorflow/contrib/py2tf/pyct/static_analysis",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "print_functions_test",
    srcs = ["print_functions_test.py"],
    deps = [
        ":convert",
        "//tensorflow/contrib/py2tf/pyct",
        "//tensorflow/contrib/py2tf/pyct/static_analysis",
        "//tensorflow/python:client_testlib",
        "@gast_archive//:gast",
    ],
)

py_test(
    name = "side_effect_guards_test",
    srcs = ["side_effect_guards_test.py"],
    deps = [
        ":convert",
        "//tensorflow/contrib/py2tf/pyct",
        "//tensorflow/contrib/py2tf/pyct/static_analysis",
        "//tensorflow/python:client_testlib",
    ],
)

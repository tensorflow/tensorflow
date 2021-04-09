load("//tensorflow:tensorflow.bzl", "py_test")

package(
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
    visibility = ["//tensorflow:__subpackages__"],
)

py_library(
    name = "operators",
    srcs = [
        "__init__.py",
        "conditional_expressions.py",
        "control_flow.py",
        "data_structures.py",
        "exceptions.py",
        "logical.py",
        "py_builtins.py",
        "slices.py",
        "variables.py",
    ],
    srcs_version = "PY3",
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:control_flow_util",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:list_ops",
        "//tensorflow/python:tensor_array_ops",
        "//tensorflow/python:tensor_util",
        "//tensorflow/python:util",
        "//tensorflow/python:variables",
        "//tensorflow/python/autograph/utils",
        "//tensorflow/python/data/experimental/ops:cardinality",
        "//tensorflow/python/data/experimental/ops:scan_ops",
        "//tensorflow/python/data/experimental/ops:take_while_ops",
        "//tensorflow/python/data/ops:dataset_ops",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "data_structures_test",
    srcs = ["data_structures_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":operators",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "conditional_expressions_test",
    srcs = ["conditional_expressions_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":operators",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "control_flow_test",
    srcs = ["control_flow_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "no_gpu",  # b/127001953
    ],
    deps = [
        ":operators",
        "//tensorflow/python:client_testlib",
        # TODO(b/145618471): Remove this transitive dependency.
        "//tensorflow/python/distribute:input_lib",
    ],
)

py_test(
    name = "exceptions_test",
    srcs = ["exceptions_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":operators",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "logical_test",
    srcs = ["logical_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":operators",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "py_builtins_test",
    srcs = ["py_builtins_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":operators",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/autograph/core",
        # TODO(b/145618471): Remove this transitive dependency.
        "//tensorflow/python/distribute:input_lib",
        "//tensorflow/python/ops/parallel_for:control_flow_ops",
        "//tensorflow/python/ops/signal",
    ],
)

py_test(
    name = "slices_test",
    srcs = ["slices_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":operators",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "variables_test",
    srcs = ["variables_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":operators",
        "//tensorflow/python:client_testlib",
    ],
)

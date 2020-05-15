load("//tensorflow:tensorflow.bzl", "py_test")

package(
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["LICENSE"])

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
        "control_flow.py",
        "control_flow_deprecated_py2.py",
        "data_structures.py",
        "exceptions.py",
        "logical.py",
        "py_builtins.py",
        "slices.py",
        "variables.py",
    ],
    srcs_version = "PY2AND3",
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
        "//tensorflow/python/data/ops:dataset_ops",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "data_structures_test",
    srcs = ["data_structures_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
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
        "no_oss_py2",
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
    srcs_version = "PY2AND3",
    deps = [
        ":operators",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "logical_test",
    srcs = ["logical_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":operators",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "py_builtins_test",
    srcs = ["py_builtins_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    tags = ["no_windows"],
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
    name = "py_builtins_py3_test",
    srcs = ["py_builtins_py3_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "no_windows",
        # TODO(kkb): Temporay workaround since KokoroPresubmit was failing.
        #                 cl/259400943 for more context.
        "no_oss_py2",
    ],
    deps = [
        ":operators",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "slices_test",
    srcs = ["slices_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":operators",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "variables_test",
    srcs = ["variables_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":operators",
        "//tensorflow/python:client_testlib",
    ],
)

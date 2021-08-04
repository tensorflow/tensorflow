# Description:
#   Python API for XLA.

package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

py_library(
    name = "types",
    srcs = ["types.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/xla:xla_data_proto_py",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:platform",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "xla_shape",
    srcs = ["xla_shape.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":types",
        "//tensorflow/compiler/xla:xla_data_proto_py",
    ],
)

py_library(
    name = "xla_literal",
    srcs = ["xla_literal.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":types",
        ":xla_shape",
        "//tensorflow/compiler/xla:xla_data_proto_py",
    ],
)

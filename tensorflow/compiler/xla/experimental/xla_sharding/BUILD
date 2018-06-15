# Description:
#   Python API for shardings in XLA.

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//tensorflow:internal"])

py_library(
    name = "xla_sharding",
    srcs = ["xla_sharding.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/xla:xla_data_proto_py",
        "//tensorflow/compiler/xla/python_api:types",
        "//tensorflow/compiler/xla/python_api:xla_shape",
        "//third_party/py/numpy",
    ],
)

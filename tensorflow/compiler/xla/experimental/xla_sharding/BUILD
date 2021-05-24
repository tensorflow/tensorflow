# Description:
#   Python API for shardings in XLA.

package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

py_library(
    name = "xla_sharding",
    srcs = ["xla_sharding.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/tf2xla/python:xla",
        "//tensorflow/compiler/xla:xla_data_proto_py",
        "//tensorflow/compiler/xla/python_api:types",
        "//tensorflow/compiler/xla/python_api:xla_shape",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "xla_sharding_test",
    srcs = ["xla_sharding_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":xla_sharding",
        "//tensorflow:tensorflow_py",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:absltest",
    ],
)

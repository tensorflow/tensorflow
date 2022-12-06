load("//tensorflow:pytype.default.bzl", "pytype_library")

pytype_library(
    name = "test_util",
    testonly = True,
    srcs = [
        "test_backend_name.py",
        "test_util.py",
    ],
    visibility = [
        "//tensorflow/dtensor:dtensor-internal",
        "//tensorflow/dtensor:dtensor-users",
    ],
    deps = [
        "//tensorflow/dtensor/python:api",
        "//tensorflow/dtensor/python:config",
        "//tensorflow/dtensor/python:layout",
        "//tensorflow/dtensor/python:tpu_util",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/flags",
        "@absl_py//absl/testing:parameterized",
    ],
)

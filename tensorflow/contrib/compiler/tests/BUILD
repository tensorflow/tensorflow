# These tests depend on contrib, so they can't be in //tensorflow/compiler/tests

load("//tensorflow/compiler/tests:build_defs.bzl", "tf_xla_py_test")

package(
    licenses = ["notice"],  # Apache 2.0
)

tf_xla_py_test(
    name = "adamax_test",
    size = "small",
    srcs = ["adamax_test.py"],
    deps = [
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/contrib/opt:opt_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:training",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_test(
    name = "addsign_test",
    size = "small",
    srcs = ["addsign_test.py"],
    deps = [
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/contrib/opt:opt_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:training",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_test(
    name = "powersign_test",
    size = "small",
    srcs = ["powersign_test.py"],
    deps = [
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/contrib/opt:opt_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:training",
        "//third_party/py/numpy",
    ],
)

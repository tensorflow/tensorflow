load("//tensorflow:tensorflow.bzl", "pytype_strict_library")
load("//tensorflow:tensorflow.bzl", "py_strict_test")

package(
    licenses = ["notice"],
)

pytype_strict_library(
    name = "type_dispatch",
    srcs = [
        "type_dispatch.py",
    ],
    srcs_version = "PY3",
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/python/types",
    ],
)

py_strict_test(
    name = "type_dispatch_test",
    srcs = ["type_dispatch_test.py"],
    python_version = "PY3",
    deps = [
        ":type_dispatch",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/types",
    ],
)

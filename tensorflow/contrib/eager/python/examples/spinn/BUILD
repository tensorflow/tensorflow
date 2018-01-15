licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//tensorflow:internal"])

load("//tensorflow:tensorflow.bzl", "cuda_py_test")
load("//tensorflow:tensorflow.bzl", "py_test")

py_library(
    name = "data",
    srcs = ["data.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = ["//third_party/py/numpy"],
)

py_test(
    name = "data_test",
    size = "small",
    srcs = ["data_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":data",
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "spinn_test",
    size = "medium",
    srcs = ["spinn_test.py"],
    additional_deps = [
        ":data",
        "//third_party/examples/eager/spinn",
        "//third_party/py/numpy",
        "//tensorflow:tensorflow_py",
        "//tensorflow/contrib/summary:summary_test_util",
        "//tensorflow/python/eager:test",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_test_lib",
    ],
    tags = ["no_pip"],  # because spinn.py is under third_party/.
)

licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "tf_py_test")

package(
    default_visibility = [
        "//tensorflow:__subpackages__",
    ],
)

py_library(
    name = "ipu",
    srcs = [
      "__init__.py",
      "python/ops.py",
      "python/utils.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/compiler/plugin/poplar:ipu_ops",
    ],
)

tf_py_test(
    name = "ops_test",
    size = "small",
    srcs = ["python/ops_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/contrib:contrib_py",
        "//tensorflow/python:framework_test_lib",
    ],
)
# Build multiplex_4 custom ops examples, which is similar to np.where

load("//tensorflow:tensorflow.bzl", "py_strict_binary", "py_strict_library")
load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")
load("//tensorflow:tensorflow.bzl", "tf_py_test")

licenses(["notice"])

tf_custom_op_library(
    name = "multiplex_4_kernel.so",
    srcs = [
        "multiplex_4_kernel.cc",
        "multiplex_4_op.cc",
    ],
)

py_strict_library(
    name = "multiplex_4_op",
    srcs = ["multiplex_4_op.py"],
    data = ["multiplex_4_kernel.so"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

py_strict_library(
    name = "model_using_multiplex",
    srcs = ["model_using_multiplex.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

tf_py_test(
    name = "multiplex_4_test",
    size = "medium",  # This test blocks because it writes and reads a file,
    timeout = "short",  # but it still runs quickly.
    srcs = ["multiplex_4_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "no_mac",  # TODO(b/216321151): Re-enable this test.
        "no_pip",
    ],
    deps = [
        ":model_using_multiplex",
        ":multiplex_4_op",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:test_lib",
        "//third_party/py/numpy",
    ],
)

py_strict_binary(
    name = "multiplex_2_save",
    srcs = ["multiplex_2_save.py"],
    srcs_version = "PY3",
    deps = [
        "model_using_multiplex",
        "//tensorflow/examples/custom_ops_doc/multiplex_2:multiplex_2_op",
        "@absl_py//absl:app",
    ],
)

py_strict_binary(
    name = "multiplex_4_load_use",
    srcs = ["multiplex_4_load_use.py"],
    srcs_version = "PY3",
    deps = [
        "model_using_multiplex",
        "multiplex_4_op",
        "@absl_py//absl:app",
    ],
)

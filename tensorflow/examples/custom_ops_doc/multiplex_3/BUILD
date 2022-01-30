# Build multiplex_3 custom ops example, which is similar to np.where.
# This example shows how a Python wrapper can choose either to use "dispach
# for custom object types" to choose an old C++ Op (that supports only dense
# tensors) for backwards compatibility or a new C++ for new functionality
# (that supprots sparse tensors).

load("//tensorflow:tensorflow.bzl", "py_strict_library")
load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")
load("//tensorflow:tensorflow.bzl", "tf_py_test")

licenses(["notice"])

tf_custom_op_library(
    name = "multiplex_3_kernel.so",
    srcs = [
        "multiplex_3_kernel.cc",
        "multiplex_3_op.cc",
    ],
)

py_strict_library(
    name = "multiplex_3_op",
    srcs = ["multiplex_3_op.py"],
    data = [":multiplex_3_kernel.so"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/examples/custom_ops_doc/multiplex_2:multiplex_2_op",
    ],
)

tf_py_test(
    name = "multiplex_3_test",
    size = "small",
    srcs = ["multiplex_3_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "no_mac",  # TODO(b/216321151): Re-enable this test.
        "no_pip",
    ],
    deps = [
        ":multiplex_3_op",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:test_lib",
        "//third_party/py/numpy",
    ],
)

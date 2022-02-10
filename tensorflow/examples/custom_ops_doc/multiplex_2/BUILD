# Build multiplex_2 custom ops example, which is similar to np.where.
# This example supports GPU (and CPU), in contrast to multiplex_1 which
# only supports CPU.

load("//tensorflow:tensorflow.bzl", "py_strict_library")
load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")
load("//tensorflow:tensorflow.bzl", "cuda_py_test")

licenses(["notice"])

tf_custom_op_library(
    name = "multiplex_2_kernel.so",
    srcs = [
        "multiplex_2_kernel.cc",
        "multiplex_2_kernel.h",
        "multiplex_2_op.cc",
    ],
    gpu_srcs = [
        "multiplex_2_kernel.h",
        "multiplex_2_kernel.cu.cc",
    ],
    visibility = ["//tensorflow/examples/custom_ops_doc:__subpackages__"],
)

py_strict_library(
    name = "multiplex_2_op",
    srcs = ["multiplex_2_op.py"],
    data = ["multiplex_2_kernel.so"],
    srcs_version = "PY3",
    visibility = ["//tensorflow/examples/custom_ops_doc:__subpackages__"],
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "multiplex_2_test",
    size = "small",
    srcs = ["multiplex_2_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "no_mac",  # TODO(b/216321151): Re-enable this test.
        "no_pip",
    ],
    deps = [
        ":multiplex_2_op",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:test_lib",
        "//third_party/py/numpy",
    ],
)

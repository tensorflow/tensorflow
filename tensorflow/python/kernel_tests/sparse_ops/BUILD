# Tests of TensorFlow sparse ops written using the Python API.

load("//tensorflow:tensorflow.bzl", "cuda_py_test")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "tf_py_test")

package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

tf_py_test(
    name = "sparse_add_op_test",
    size = "small",
    srcs = ["sparse_add_op_test.py"],
    deps = [
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:sparse_grad",
        "//tensorflow/python:sparse_ops",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "sparse_concat_op_test",
    size = "small",
    srcs = ["sparse_concat_op_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:sparse_ops",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "sparse_conditional_accumulator_test",
    size = "small",
    srcs = ["sparse_conditional_accumulator_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:data_flow_ops",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "sparse_cross_op_test",
    size = "small",
    srcs = ["sparse_cross_op_test.py"],
    tags = ["no_windows"],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:sparse_ops",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "sparse_matmul_op_test",
    size = "medium",
    srcs = ["sparse_matmul_op_test.py"],
    tags = ["no_windows"],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "sparse_ops_test",
    size = "medium",
    srcs = ["sparse_ops_test.py"],
    shard_count = 5,
    tags = [
        "optonly",  # b/77589990
    ],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:sparse_grad",
        "//tensorflow/python:sparse_ops",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "sparse_reorder_op_test",
    size = "small",
    srcs = ["sparse_reorder_op_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:sparse_grad",
        "//tensorflow/python:sparse_ops",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "sparse_reshape_op_test",
    size = "small",
    srcs = ["sparse_reshape_op_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:sparse_ops",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "sparse_serialization_ops_test",
    size = "small",
    srcs = ["sparse_serialization_ops_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:sparse_ops",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "sparse_slice_op_test",
    size = "small",
    srcs = ["sparse_slice_op_test.py"],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:sparse_grad",
        "//tensorflow/python:sparse_ops",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "sparse_split_op_test",
    size = "small",
    srcs = ["sparse_split_op_test.py"],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:sparse_ops",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "sparse_tensor_dense_matmul_grad_test",
    size = "small",
    srcs = ["sparse_tensor_dense_matmul_grad_test.py"],
    xla_tags = [
        "no_cuda_asan",  # b/182392418 times out
    ],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:sparse_grad",
        "//tensorflow/python:sparse_ops",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    # Test name is shortened to workaround Windows Bazel's 259 char path limit.
    name = "sparse_tensor_dense_matmul_op_d9m_test",
    size = "small",
    srcs = ["sparse_tensor_dense_matmul_op_d9m_test.py"],
    tags = [
        "no_cuda_asan",
        "noasan",
        "notsan",
    ],  # b/192492331
    xla_tags = [
        "no_cuda_asan",  # times out
    ],  # b/192492331
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:errors",
        "//tensorflow/python:sparse_ops",
        "//tensorflow/python:sparse_tensor",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "sparse_tensor_dense_matmul_op_test",
    size = "medium",
    srcs = ["sparse_tensor_dense_matmul_op_test.py"],
    tags = [
        "notap",  # b/231313017
    ],
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:sparse_ops",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "sparse_tensors_map_ops_test",
    size = "small",
    srcs = ["sparse_tensors_map_ops_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:sparse_ops",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "sparse_to_dense_op_py_test",
    size = "small",
    srcs = ["sparse_to_dense_op_py_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:sparse_ops",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "sparse_xent_op_d9m_test",
    size = "medium",
    srcs = ["sparse_xent_op_d9m_test.py"],
    tags = ["no_windows"],  # Fails as SegmentSum is nondeterministic on Windows
    xla_enable_strict_auto_jit = False,
    deps = [
        ":sparse_xent_op_test_base",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:nn_grad",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:nn_ops_gen",
        "//tensorflow/python/eager:backprop",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "sparse_xent_op_test",
    size = "small",
    srcs = ["sparse_xent_op_test.py"],
    deps = [
        ":sparse_xent_op_test_base",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:gradients",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_grad",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:sparse_ops",
        "//third_party/py/numpy",
        "@absl_py//absl:app",
    ],
)

py_library(
    name = "sparse_xent_op_test_base",
    srcs = ["sparse_xent_op_test_base.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:nn_grad",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:context",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "sparsemask_op_test",
    size = "small",
    srcs = ["sparsemask_op_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//third_party/py/numpy",
    ],
)

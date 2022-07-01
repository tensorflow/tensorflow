# Tests of TensorFlow math kernels written using the Python API.

load("//tensorflow:tensorflow.bzl", "cuda_py_test")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "tf_py_test")

package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

cuda_py_test(
    name = "aggregate_ops_test",
    size = "small",
    srcs = ["aggregate_ops_test.py"],
    xla_tags = [
        "no_cuda_asan",  # times out
    ],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:string_ops",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "approx_topk_test",
    srcs = ["approx_topk_test.py"],
    xla_enable_strict_auto_jit = True,
    xla_enabled = True,
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:nn_ops",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "argmax_op_test",
    size = "small",
    srcs = ["argmax_op_test.py"],
    tags = [
        "no_cuda_asan",  # times out
    ],
    xla_tags = [
        "no_cuda_asan",  # times out
    ],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:math_ops",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "banded_triangular_solve_op_test",
    size = "small",
    srcs = ["banded_triangular_solve_op_test.py"],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:linalg_ops",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "basic_gpu_test",
    size = "small",
    srcs = ["basic_gpu_test.py"],
    grpc_enabled = True,
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:gradient_checker_v2",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_test(
    name = "batch_matmul_op_test",
    size = "medium",
    srcs = ["batch_matmul_op_test.py"],
    shard_count = 20,
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "bincount_op_test",
    size = "small",
    srcs = ["bincount_op_test.py"],
    tags = ["no_windows_gpu"],
    xla_enable_strict_auto_jit = True,
    deps = [
        "//tensorflow/python:bincount_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
    ],
)

cuda_py_test(
    name = "bucketize_op_test",
    size = "medium",
    srcs = ["bucketize_op_test.py"],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "clip_ops_test",
    size = "small",
    srcs = ["clip_ops_test.py"],
    tags = [
        "no_gpu",  # b/127001953
        "no_windows",
    ],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:clip_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
    ],
)

tf_py_test(
    name = "confusion_matrix_test",
    size = "small",
    srcs = ["confusion_matrix_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:confusion_matrix",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_ops",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "cross_grad_test",
    size = "small",
    srcs = ["cross_grad_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:math_ops",
    ],
)

cuda_py_test(
    name = "cumulative_logsumexp_test",
    size = "medium",
    srcs = ["cumulative_logsumexp_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:map_fn",
        "//tensorflow/python:math_ops",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "cwise_ops_test",
    size = "medium",
    srcs = ["cwise_ops_test.py"],
    shard_count = 50,
    tags = [
        "no_cuda_asan",  # b/179032113
    ],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:gradients",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:math_ops_gen",
        "//tensorflow/python:nn_grad",
        "//tensorflow/python:platform",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "cwise_ops_binary_test",
    size = "medium",
    srcs = ["cwise_ops_binary_test.py"],
    shard_count = 50,
    # b/140155647: Error just outside of tolerance
    xla_enable_strict_auto_jit = False,
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:gradients",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:math_ops_gen",
        "//tensorflow/python:nn_grad",
        "//tensorflow/python:platform",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "cwise_ops_unary_test",
    size = "medium",
    srcs = ["cwise_ops_unary_test.py"],
    shard_count = 50,
    tags = [
        "no_windows",  # TODO(b/207048097): re-enable
    ],
    # b/140155706: nans in result
    xla_enable_strict_auto_jit = False,
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:gradients",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:math_ops_gen",
        "//tensorflow/python:nn_grad",
        "//tensorflow/python:platform",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "division_future_test",
    size = "medium",
    srcs = ["division_future_test.py"],
    tags = ["manual"],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "division_past_test",
    size = "medium",
    srcs = ["division_past_test.py"],
    tags = [
        "manual",
        "no_windows",
    ],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "in_topk_op_test",
    size = "small",
    srcs = ["in_topk_op_test.py"],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:nn_ops",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "matmul_op_test",
    size = "medium",
    srcs = ["matmul_op_test.py"],
    shard_count = 20,
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "reduce_benchmark_test",
    srcs = ["reduce_benchmark_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:gradients",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:platform_benchmark",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:context",
    ],
)

cuda_py_test(
    name = "reduction_ops_test_big",
    size = "medium",
    srcs = ["reduction_ops_test_big.py"],
    shard_count = 3,
    tags = [
        "manual",
        "no_gpu",
        "noguitar",
    ],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "reduction_ops_test",
    size = "medium",
    srcs = ["reduction_ops_test.py"],
    shard_count = 6,
    tags = [
        "no_cuda_asan",  # times out
        "no_windows_gpu",
    ],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "segment_reduction_ops_d9m_test",
    size = "small",
    srcs = ["segment_reduction_ops_d9m_test.py"],
    tags = ["no_cuda_asan"],  # b/195247538
    xla_enable_strict_auto_jit = False,
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:backprop",
    ],
)

cuda_py_test(
    name = "segment_reduction_ops_test",
    size = "medium",
    srcs = ["segment_reduction_ops_test.py"],
    shard_count = 10,
    # TODO (b/173835746): the test fails with XLA.
    xla_enable_strict_auto_jit = False,
    deps = [
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_grad",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "sets_test",
    size = "medium",
    srcs = ["sets_test.py"],
    deps = [
        "//tensorflow/python:errors",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:sets",
        "//tensorflow/python:sparse_ops",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "tensordot_op_test",
    size = "medium",
    srcs = ["tensordot_op_test.py"],
    shard_count = 20,
    tags = ["no_rocm"],
    xla_enable_strict_auto_jit = False,  # b/161856380
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "topk_op_test",
    size = "medium",
    srcs = ["topk_op_test.py"],
    xla_tags = [
        "no_cuda_asan",  # times out
    ],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:gradients",
        "//tensorflow/python:nn_grad",
        "//tensorflow/python:nn_ops",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "transpose_op_test",
    size = "medium",
    srcs = ["transpose_op_test.py"],
    shard_count = 10,
    tags = [
        "no_gpu",
        "no_oss",
        "notap",  # flaky timeout/segfault, b/136498892
        "optonly",  # times out
    ],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "zero_division_test",
    size = "medium",
    srcs = ["zero_division_test.py"],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
    ],
)

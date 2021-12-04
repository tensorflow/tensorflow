# Tests of TensorFlow kernels written using the Python API.

load("//tensorflow:tensorflow.bzl", "cuda_py_test")
load("//tensorflow:tensorflow.bzl", "tf_py_test")

package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

# CPU only tests should use tf_py_test, GPU tests use cuda_py_test
# Please avoid the py_tests and cuda_py_tests (plural) while we
# fix the shared/overbroad dependencies.

tf_py_test(
    name = "candidate_sampler_ops_test",
    size = "small",
    srcs = ["candidate_sampler_ops_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:candidate_sampling_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "multinomial_op_big_test",
    size = "medium",
    srcs = ["multinomial_op_big_test.py"],
    shard_count = 3,
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_ops",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "multinomial_op_test",
    size = "small",
    srcs = ["multinomial_op_test.py"],
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_ops",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "parameterized_truncated_normal_op_test",
    size = "medium",
    srcs = ["parameterized_truncated_normal_op_test.py"],
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:platform",
        "//tensorflow/python:random_ops",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

# TODO(b/130359919): Reenable test when it becomes stable
tf_py_test(
    name = "random_binomial_test",
    size = "medium",
    srcs = ["random_binomial_test.py"],
    shard_count = 3,
    tags = ["no_oss"],
    deps = [
        ":util",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:platform",
        "//tensorflow/python:stateful_random_ops",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "random_crop_test",
    size = "small",
    srcs = ["random_crop_test.py"],
    xla_tags = [
        "no_cuda_asan",  # times out
    ],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:random_ops",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "random_gamma_test",
    size = "medium",
    srcs = ["random_gamma_test.py"],
    shard_count = 4,
    tags = ["nozapfhahn"],
    xla_tags = [
        "no_cuda_asan",  # times out
    ],
    deps = [
        ":util",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:platform",
        "//tensorflow/python:random_ops",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "random_grad_test",
    size = "small",
    srcs = ["random_grad_test.py"],
    xla_tags = [
        "no_cuda_asan",  # times out
    ],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:random_grad",
        "//tensorflow/python:random_ops",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "random_ops_test",
    size = "medium",
    srcs = ["random_ops_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:random_ops",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "random_poisson_test",
    size = "medium",
    srcs = ["random_poisson_test.py"],
    deps = [
        ":util",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:platform",
        "//tensorflow/python:random_ops",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "random_shuffle_queue_test",
    size = "small",
    srcs = ["random_shuffle_queue_test.py"],
    tags = [
        "no_cuda_on_cpu_tap",  # TODO(b/171060960) flakyly broken assertions
    ],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:data_flow_ops",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:platform",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "stateless_random_ops_test",
    size = "medium",
    srcs = ["stateless_random_ops_test.py"],
    shard_count = 10,
    xla_tags = [
        "no_cuda_asan",  # times-out
    ],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:stateless_random_ops",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "util",
    srcs = ["util.py"],
    srcs_version = "PY3",
    deps = [
        "//third_party/py/numpy",
    ],
)

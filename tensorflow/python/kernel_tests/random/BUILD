# Tests of TensorFlow kernels written using the Python API.

load("//tensorflow:strict.default.bzl", "py_strict_library")
load("//tensorflow:tensorflow.default.bzl", "cuda_py_strict_test", "tf_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

# CPU only tests should use tf_py_test, GPU tests use cuda_py_test
# Please avoid the py_tests and cuda_py_tests (plural) while we
# fix the shared/overbroad dependencies.

tf_py_strict_test(
    name = "candidate_sampler_ops_test",
    size = "small",
    srcs = ["candidate_sampler_ops_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:candidate_sampling_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:test_lib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "multinomial_op_big_test",
    size = "medium",
    srcs = ["multinomial_op_big_test.py"],
    shard_count = 3,
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:random_ops",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:random_seed",
        "//tensorflow/python/framework:test_lib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "multinomial_op_test",
    size = "small",
    srcs = ["multinomial_op_test.py"],
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python/client:session",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:random_seed",
        "//tensorflow/python/framework:test_lib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "parameterized_truncated_normal_op_test",
    size = "medium",
    srcs = ["parameterized_truncated_normal_op_test.py"],
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:stateless_random_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/client:session",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/framework:random_seed",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:tf_logging",
        "//third_party/py/numpy",
    ],
)

# TODO(b/130359919): Reenable test when it becomes stable
tf_py_strict_test(
    name = "random_binomial_test",
    size = "medium",
    srcs = ["random_binomial_test.py"],
    shard_count = 3,
    tags = ["no_oss"],
    deps = [
        ":util",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:stateful_random_ops",
        "//tensorflow/python:stateless_random_ops",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:tf_logging",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "random_crop_test",
    size = "small",
    srcs = ["random_crop_test.py"],
    xla_tags = [
        "no_cuda_asan",  # times out
    ],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:random_crop_ops",
        "//tensorflow/python/framework:test_lib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "random_gamma_test",
    size = "medium",
    srcs = ["random_gamma_test.py"],
    shard_count = 4,
    tags = [
        "no_windows",  # TODO(b/218876472)
        "nozapfhahn",
    ],
    xla_tags = [
        "no_cuda_asan",  # times out
        "nomsan",  # b/216320808
    ],
    deps = [
        ":util",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:random_seed",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:tf_logging",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "random_grad_test",
    size = "small",
    srcs = ["random_grad_test.py"],
    xla_tags = [
        "no_cuda_asan",  # times out
    ],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:gradients_impl",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_grad",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:stateless_random_ops_v2_gen",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:tf_logging",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "random_ops_test",
    size = "medium",
    srcs = ["random_ops_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:random_ops_gen",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:random_seed",
        "//tensorflow/python/framework:test_lib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "random_poisson_test",
    size = "medium",
    srcs = ["random_poisson_test.py"],
    deps = [
        ":util",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:random_ops",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:tf_logging",
        "//third_party/py/numpy",
    ],
)

tf_py_strict_test(
    name = "random_shuffle_queue_test",
    size = "medium",
    srcs = ["random_shuffle_queue_test.py"],
    tags = [
        "no_cuda_on_cpu_tap",  # TODO(b/171060960) flakyly broken assertions
    ],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:data_flow_ops",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:random_seed",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:tf_logging",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "stateless_random_ops_test",
    size = "medium",
    srcs = ["stateless_random_ops_test.py"],
    shard_count = 10,
    xla_tags = [
        "no_cuda_asan",  # times-out
    ],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:array_ops_stack",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:stateless_random_ops",
        "//tensorflow/python:stateless_random_ops_v2_gen",
        "//tensorflow/python/compat",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:random_seed",
        "//tensorflow/python/framework:test_lib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "stateful_random_ops_test",
    size = "medium",
    srcs = ["stateful_random_ops_test.py"],
    python_version = "PY3",
    xla_enable_strict_auto_jit = False,
    xla_enabled = True,
    deps = [
        ":util",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:logging_ops",
        "//tensorflow/python:random_ops_gen",
        "//tensorflow/python:stateful_random_ops",
        "//tensorflow/python:stateful_random_ops_gen",
        "//tensorflow/python:variables",
        "//tensorflow/python/checkpoint",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:test_lib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "random_index_shuffle_test",
    srcs = ["random_index_shuffle_test.py"],
    shard_count = 10,
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_index_shuffle_ops_gen",
        "//tensorflow/python:stateless_random_ops",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_library(
    name = "util",
    srcs = ["util.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/ops/distributions",
        "//third_party/py/numpy",
    ],
)

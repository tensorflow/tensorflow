# Tests of TensorFlow kernels written using the Python API.

package(
    default_visibility = ["//tensorflow:internal"],
)

licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "cuda_py_test")

cuda_py_test(
    name = "bijector_test",
    size = "small",
    srcs = ["bijector_test.py"],
    additional_deps = [
        "//tensorflow/python/ops/distributions",
        "//third_party/py/numpy",
        "@six_archive//:six",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
    ],
    xla_enable_strict_auto_jit = True,
)

cuda_py_test(
    name = "util_test",
    size = "medium",
    srcs = ["util_test.py"],
    additional_deps = [
        "//tensorflow/python/ops/distributions",
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:gradients",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
    ],
    shard_count = 3,
    xla_enable_strict_auto_jit = True,
)

cuda_py_test(
    name = "kullback_leibler_test",
    size = "small",
    srcs = ["kullback_leibler_test.py"],
    additional_deps = [
        "//tensorflow/python/ops/distributions",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:platform_test",
    ],
    xla_enable_strict_auto_jit = True,
)

cuda_py_test(
    name = "beta_test",
    size = "small",
    srcs = ["beta_test.py"],
    additional_deps = [
        "//tensorflow/python/ops/distributions",
        "//third_party/py/numpy",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:platform_test",
    ],
    xla_enable_strict_auto_jit = True,
)

cuda_py_test(
    name = "bernoulli_test",
    size = "small",
    srcs = ["bernoulli_test.py"],
    additional_deps = [
        "//tensorflow/python/ops/distributions",
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
    ],
    xla_enable_strict_auto_jit = True,
)

cuda_py_test(
    name = "categorical_test",
    size = "small",
    srcs = ["categorical_test.py"],
    additional_deps = [
        "@absl_py//absl/testing:parameterized",
        "//tensorflow/python/ops/distributions",
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:random_ops",
    ],
    xla_enable_strict_auto_jit = True,
)

cuda_py_test(
    name = "dirichlet_test",
    size = "small",
    srcs = ["dirichlet_test.py"],
    additional_deps = [
        "//tensorflow/python/ops/distributions",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
    xla_enable_strict_auto_jit = True,
)

cuda_py_test(
    name = "dirichlet_multinomial_test",
    size = "medium",
    srcs = ["dirichlet_multinomial_test.py"],
    additional_deps = [
        "//tensorflow/python/ops/distributions",
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
    ],
    tags = [
        "noguitar",  # b/110489471
        "notap",  # b/110489471
    ],
    xla_enable_strict_auto_jit = True,
)

cuda_py_test(
    name = "exponential_test",
    srcs = ["exponential_test.py"],
    additional_deps = [
        "//tensorflow/python/ops/distributions",
        "//third_party/py/numpy",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:platform_test",
    ],
    xla_enable_strict_auto_jit = True,
)

cuda_py_test(
    name = "gamma_test",
    srcs = ["gamma_test.py"],
    additional_deps = [
        "//tensorflow/python/ops/distributions",
        "//third_party/py/numpy",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:platform_test",
    ],
    xla_enable_strict_auto_jit = True,
)

cuda_py_test(
    name = "laplace_test",
    srcs = ["laplace_test.py"],
    additional_deps = [
        "//tensorflow/python/ops/distributions",
        "//third_party/py/numpy",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:platform_test",
    ],
    xla_enable_strict_auto_jit = True,
)

cuda_py_test(
    name = "multinomial_test",
    srcs = ["multinomial_test.py"],
    additional_deps = [
        "//tensorflow/python/ops/distributions",
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
    ],
    tags = ["manual"],  # b/69001419
    xla_enable_strict_auto_jit = True,
)

cuda_py_test(
    name = "student_t_test",
    size = "small",
    srcs = ["student_t_test.py"],
    additional_deps = [
        "//tensorflow/python/ops/distributions",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:platform_test",
    ],
    tags = [
        # TODO(b/121223043): Re-enable this test after fixing "mean not defined"
        # errors.
        "no_mac",
        "no_oss",
        # disable to avoid false positives from scipy.
        "nomsan",
    ],
    xla_enable_strict_auto_jit = True,
)

cuda_py_test(
    name = "uniform_test",
    size = "small",
    srcs = ["uniform_test.py"],
    additional_deps = [
        "//tensorflow/python/ops/distributions",
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
    ],
    xla_enable_strict_auto_jit = True,
)

cuda_py_test(
    name = "normal_test",
    size = "medium",
    srcs = ["normal_test.py"],
    additional_deps = [
        "//tensorflow/python/ops/distributions",
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:gradients",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:variables",
    ],
    xla_enable_strict_auto_jit = True,
)

cuda_py_test(
    name = "special_math_test",
    size = "medium",
    srcs = ["special_math_test.py"],
    additional_deps = [
        "//tensorflow/python/ops/distributions",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:gradients",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:variables",
    ],
    xla_enable_strict_auto_jit = True,
)

cuda_py_test(
    name = "identity_bijector_test",
    size = "small",
    srcs = ["identity_bijector_test.py"],
    additional_deps = [
        "//tensorflow/python/ops/distributions",
        "//third_party/py/numpy",
        "@six_archive//:six",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
    ],
    xla_enable_strict_auto_jit = True,
)

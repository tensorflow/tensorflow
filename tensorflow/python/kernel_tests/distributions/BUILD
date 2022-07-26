# Tests of TensorFlow kernels written using the Python API.

load("//tensorflow:tensorflow.bzl", "cuda_py_test")

package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

cuda_py_test(
    name = "bijector_test",
    size = "small",
    srcs = ["bijector_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/ops/distributions",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

cuda_py_test(
    name = "util_test",
    size = "medium",
    srcs = ["util_test.py"],
    shard_count = 3,
    xla_enable_strict_auto_jit = False,  # TODO(b/144920376)
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:gradients",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/ops/distributions",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "kullback_leibler_test",
    size = "small",
    srcs = ["kullback_leibler_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/ops/distributions",
    ],
)

cuda_py_test(
    name = "beta_test",
    size = "small",
    srcs = ["beta_test.py"],
    tags = [
        "no_oss",  # b/205574093, and because tf.distributions is deprecated
        "nomac",  # b/191763315
        "notsan",  # b/173653918
    ],
    xla_tags = [
        "no_cuda_asan",  # times out
    ],
    deps = [
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/ops/distributions",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "bernoulli_test",
    size = "medium",
    srcs = ["bernoulli_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/ops/distributions",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "categorical_test",
    size = "small",
    srcs = ["categorical_test.py"],
    xla_tags = [
        "no_cuda_asan",  # times out
    ],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:random_ops",
        "//tensorflow/python/ops/distributions",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_test(
    name = "dirichlet_test",
    size = "small",
    srcs = ["dirichlet_test.py"],
    xla_tags = [
        "no_cuda_asan",  # times out
    ],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/ops/distributions",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "dirichlet_multinomial_test",
    size = "medium",
    srcs = ["dirichlet_multinomial_test.py"],
    tags = [
        "noguitar",  # b/110489471
        "notap",  # b/110489471
    ],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/ops/distributions",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "exponential_test",
    srcs = ["exponential_test.py"],
    deps = [
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/ops/distributions",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "gamma_test",
    srcs = ["gamma_test.py"],
    deps = [
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/ops/distributions",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "laplace_test",
    srcs = ["laplace_test.py"],
    deps = [
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/ops/distributions",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "multinomial_test",
    srcs = ["multinomial_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/ops/distributions",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "student_t_test",
    size = "small",
    srcs = ["student_t_test.py"],
    tags = [
        # TODO(b/121223043): Re-enable this test after fixing "mean not defined"
        # errors.
        "no_mac",
        "no_oss",
        # disable to avoid false positives from scipy.
        "nomsan",
        "noasan",  #b/168810473
    ],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/ops/distributions",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "uniform_test",
    size = "small",
    srcs = ["uniform_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/ops/distributions",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "normal_test",
    size = "medium",
    srcs = ["normal_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:gradients",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:variables",
        "//tensorflow/python/ops/distributions",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "special_math_test",
    size = "medium",
    srcs = ["special_math_test.py"],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:gradients",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:variables",
        "//tensorflow/python/ops/distributions",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "identity_bijector_test",
    size = "small",
    srcs = ["identity_bijector_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/ops/distributions",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

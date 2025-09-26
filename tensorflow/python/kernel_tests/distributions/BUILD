# Tests of TensorFlow kernels written using the Python API.

load("//tensorflow:tensorflow.default.bzl", "cuda_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

cuda_py_strict_test(
    name = "bijector_test",
    size = "small",
    srcs = ["bijector_test.py"],
    deps = [
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops/distributions:bijector",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "util_test",
    size = "medium",
    srcs = ["util_test.py"],
    shard_count = 3,
    deps = [
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:gradient_checker",
        "//tensorflow/python/ops:gradients_impl",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:nn_grad",
        "//tensorflow/python/ops:nn_ops",
        "//tensorflow/python/ops/distributions:util",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:tf_logging",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "kullback_leibler_test",
    size = "small",
    srcs = ["kullback_leibler_test.py"],
    deps = [
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops/distributions:kullback_leibler",
        "//tensorflow/python/ops/distributions:normal",
        "//tensorflow/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "beta_test",
    size = "small",
    srcs = ["beta_test.py"],
    tags = [
        "no_oss",  # b/205574093, and because tf.distributions is deprecated
        "nomac",  # b/191763315
    ],
    xla_tags = [
        "no_cuda_asan",  # times out
    ],
    deps = [
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/framework:random_seed",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:nn_ops",
        "//tensorflow/python/ops/distributions:beta",
        "//tensorflow/python/ops/distributions:kullback_leibler",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:tf_logging",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "bernoulli_test",
    size = "medium",
    srcs = ["bernoulli_test.py"],
    deps = [
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops/distributions:bernoulli",
        "//tensorflow/python/ops/distributions:kullback_leibler",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:tf_logging",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "categorical_test",
    size = "small",
    srcs = ["categorical_test.py"],
    xla_tags = [
        "no_cuda_asan",  # times out
    ],
    deps = [
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:gradients_impl",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:nn_ops",
        "//tensorflow/python/ops:random_ops",
        "//tensorflow/python/ops/distributions:categorical",
        "//tensorflow/python/ops/distributions:kullback_leibler",
        "//tensorflow/python/ops/distributions:normal",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "dirichlet_test",
    size = "small",
    srcs = ["dirichlet_test.py"],
    xla_tags = [
        "no_cuda_asan",  # times out
    ],
    deps = [
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops/distributions:dirichlet",
        "//tensorflow/python/ops/distributions:kullback_leibler",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:tf_logging",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "dirichlet_multinomial_test",
    size = "medium",
    srcs = ["dirichlet_multinomial_test.py"],
    tags = [
        "noguitar",  # b/110489471
        "notap",  # b/110489471
    ],
    deps = [
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops/distributions:dirichlet_multinomial",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "exponential_test",
    srcs = ["exponential_test.py"],
    deps = [
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:nn_ops",
        "//tensorflow/python/ops/distributions:exponential",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:tf_logging",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "gamma_test",
    srcs = ["gamma_test.py"],
    deps = [
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:nn_ops",
        "//tensorflow/python/ops/distributions:gamma",
        "//tensorflow/python/ops/distributions:kullback_leibler",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:tf_logging",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "laplace_test",
    srcs = ["laplace_test.py"],
    deps = [
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:nn_ops",
        "//tensorflow/python/ops/distributions:laplace",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:tf_logging",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "multinomial_test",
    srcs = ["multinomial_test.py"],
    deps = [
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops/distributions:multinomial",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "student_t_test",
    size = "small",
    srcs = ["student_t_test.py"],
    tags = [
        # TODO(b/121223043): Re-enable this test after fixing "mean not defined"
        # errors.
        "no_mac",
        "no_oss",
    ],
    deps = [
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:random_seed",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:nn_ops",
        "//tensorflow/python/ops/distributions:student_t",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:tf_logging",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "uniform_test",
    size = "small",
    srcs = ["uniform_test.py"],
    deps = [
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops_stack",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops/distributions:uniform",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:tf_logging",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "normal_test",
    size = "medium",
    srcs = ["normal_test.py"],
    deps = [
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:gradients_impl",
        "//tensorflow/python/ops:nn_ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/ops/distributions:kullback_leibler",
        "//tensorflow/python/ops/distributions:normal",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:tf_logging",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "special_math_test",
    size = "medium",
    srcs = ["special_math_test.py"],
    deps = [
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:gradient_checker",
        "//tensorflow/python/ops:gradients_impl",
        "//tensorflow/python/ops/distributions:special_math",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:tf_logging",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "identity_bijector_test",
    size = "small",
    srcs = ["identity_bijector_test.py"],
    deps = [
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops/distributions:bijector_test_util",
        "//tensorflow/python/ops/distributions:identity_bijector",
        "//tensorflow/python/platform:client_testlib",
    ],
)

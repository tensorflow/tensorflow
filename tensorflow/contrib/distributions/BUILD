# Description:
#   Contains ops for statistical distributions (with pdf, cdf, sample, etc...).
#   APIs here are meant to evolve over time.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

load("//tensorflow:tensorflow.bzl", "cuda_py_tests")

cuda_py_tests(
    name = "distribution_test",
    size = "small",
    srcs = ["python/kernel_tests/distribution_test.py"],
    additional_deps = [
        ":distributions_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_tests(
    name = "operator_pd_test",
    size = "small",
    srcs = ["python/kernel_tests/operator_pd_test.py"],
    additional_deps = [
        ":distributions_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_tests(
    name = "operator_pd_cholesky_test",
    size = "medium",
    srcs = ["python/kernel_tests/operator_pd_cholesky_test.py"],
    additional_deps = [
        ":distributions_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_tests(
    name = "operator_pd_diag_test",
    size = "medium",
    srcs = ["python/kernel_tests/operator_pd_diag_test.py"],
    additional_deps = [
        ":distributions_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_tests(
    name = "operator_pd_full_test",
    size = "medium",
    srcs = ["python/kernel_tests/operator_pd_full_test.py"],
    additional_deps = [
        ":distributions_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_tests(
    name = "operator_pd_identity_test",
    size = "medium",
    srcs = ["python/kernel_tests/operator_pd_identity_test.py"],
    additional_deps = [
        ":distributions_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_tests(
    name = "operator_pd_vdvt_update_test",
    size = "large",
    srcs = ["python/kernel_tests/operator_pd_vdvt_update_test.py"],
    additional_deps = [
        ":distributions_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
    shard_count = 5,
    tags = ["notap"],  # http://b/30441813
)

py_library(
    name = "distributions_py",
    srcs = ["__init__.py"] + glob(["python/ops/*.py"]),
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/framework:framework_py",
        "//tensorflow/contrib/linalg:linalg_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:check_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:data_flow_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:linalg_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:special_math_ops",
    ],
)

cuda_py_tests(
    name = "bernoulli_test",
    size = "small",
    srcs = ["python/kernel_tests/bernoulli_test.py"],
    additional_deps = [
        ":distributions_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_tests(
    name = "beta_test",
    size = "small",
    srcs = ["python/kernel_tests/beta_test.py"],
    additional_deps = [
        ":distributions_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_tests(
    name = "binomial_test",
    size = "small",
    srcs = ["python/kernel_tests/binomial_test.py"],
    additional_deps = [
        ":distributions_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_tests(
    name = "categorical_test",
    size = "small",
    srcs = ["python/kernel_tests/categorical_test.py"],
    additional_deps = [
        ":distributions_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_tests(
    name = "chi2_test",
    srcs = ["python/kernel_tests/chi2_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_tests(
    name = "dirichlet_test",
    size = "small",
    srcs = ["python/kernel_tests/dirichlet_test.py"],
    additional_deps = [
        ":distributions_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_tests(
    name = "dirichlet_multinomial_test",
    size = "small",
    srcs = ["python/kernel_tests/dirichlet_multinomial_test.py"],
    additional_deps = [
        ":distributions_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_tests(
    name = "exponential_test",
    srcs = ["python/kernel_tests/exponential_test.py"],
    additional_deps = [
        ":distributions_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_tests(
    name = "gamma_test",
    srcs = ["python/kernel_tests/gamma_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_tests(
    name = "inverse_gamma_test",
    srcs = ["python/kernel_tests/inverse_gamma_test.py"],
    additional_deps = [
        ":distributions_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_tests(
    name = "laplace_test",
    srcs = ["python/kernel_tests/laplace_test.py"],
    additional_deps = [
        ":distributions_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_tests(
    name = "multinomial_test",
    srcs = ["python/kernel_tests/multinomial_test.py"],
    additional_deps = [
        ":distributions_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_tests(
    name = "mvn_test",
    size = "small",
    srcs = ["python/kernel_tests/mvn_test.py"],
    additional_deps = [
        ":distributions_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_tests(
    name = "mixture_test",
    size = "medium",
    srcs = ["python/kernel_tests/mixture_test.py"],
    additional_deps = [
        ":distributions_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_tests(
    name = "normal_test",
    size = "medium",
    srcs = ["python/kernel_tests/normal_test.py"],
    additional_deps = [
        ":distributions_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_tests(
    name = "poisson_test",
    size = "small",
    srcs = ["python/kernel_tests/poisson_test.py"],
    additional_deps = [
        ":distributions_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_tests(
    name = "student_t_test",
    size = "small",
    srcs = ["python/kernel_tests/student_t_test.py"],
    additional_deps = [
        ":distributions_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
    tags = ["nomsan"],  # disable to avoid false positives from scipy.
)

cuda_py_tests(
    name = "uniform_test",
    size = "small",
    srcs = ["python/kernel_tests/uniform_test.py"],
    additional_deps = [
        ":distributions_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ],
)

cuda_py_tests(
    name = "wishart_test",
    size = "small",
    srcs = ["python/kernel_tests/wishart_test.py"],
    additional_deps = [
        ":distributions_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_tests(
    name = "kullback_leibler_test",
    size = "small",
    srcs = ["python/kernel_tests/kullback_leibler_test.py"],
    additional_deps = [
        ":distributions_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_tests(
    name = "normal_conjugate_posteriors_test",
    size = "small",
    srcs = ["python/kernel_tests/normal_conjugate_posteriors_test.py"],
    additional_deps = [
        ":distributions_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_tests(
    name = "quantized_distribution_test",
    size = "medium",
    srcs = ["python/kernel_tests/quantized_distribution_test.py"],
    additional_deps = [
        ":distributions_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_tests(
    name = "transformed_distribution_test",
    size = "medium",
    srcs = ["python/kernel_tests/transformed_distribution_test.py"],
    additional_deps = [
        ":distributions_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_tests(
    name = "distribution_util_test",
    size = "small",
    srcs = ["python/kernel_tests/distribution_util_test.py"],
    additional_deps = [
        ":distributions_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_tests(
    name = "shape_test",
    size = "small",
    srcs = ["python/kernel_tests/shape_test.py"],
    additional_deps = [
        ":distributions_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_tests(
    name = "bijector_test",
    size = "medium",
    srcs = ["python/kernel_tests/bijector_test.py"],
    additional_deps = [
        ":distributions_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
    shard_count = 5,
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)

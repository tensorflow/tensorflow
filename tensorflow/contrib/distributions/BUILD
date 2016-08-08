# Description:
#   Contains ops for statistical distributions (with pdf, cdf, sample, etc...).
#   APIs here are meant to evolve over time.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

load("//tensorflow:tensorflow.bzl", "cuda_py_tests")

cuda_py_tests(
    name = "operator_pd_test",
    size = "small",
    srcs = ["python/kernel_tests/operator_pd_test.py"],
    additional_deps = [
        ":distributions_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_tests(
    name = "operator_pd_cholesky_test",
    size = "small",
    srcs = ["python/kernel_tests/operator_pd_cholesky_test.py"],
    additional_deps = [
        ":distributions_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_tests(
    name = "operator_pd_diag_test",
    size = "small",
    srcs = ["python/kernel_tests/operator_pd_diag_test.py"],
    additional_deps = [
        ":distributions_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_tests(
    name = "operator_pd_full_test",
    size = "small",
    srcs = ["python/kernel_tests/operator_pd_full_test.py"],
    additional_deps = [
        ":distributions_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_tests(
    name = "operator_pd_identity_test",
    size = "small",
    srcs = ["python/kernel_tests/operator_pd_identity_test.py"],
    additional_deps = [
        ":distributions_py",
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
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
    tags = ["notap"],  # http://b/30441813
)

py_library(
    name = "distributions_py",
    srcs = ["__init__.py"] + glob(["python/ops/*.py"]),
    srcs_version = "PY2AND3",
)

cuda_py_tests(
    name = "bernoulli_test",
    size = "small",
    srcs = ["python/kernel_tests/bernoulli_test.py"],
    additional_deps = [
        ":distributions_py",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_tests(
    name = "beta_test",
    size = "small",
    srcs = ["python/kernel_tests/beta_test.py"],
    additional_deps = [
        ":distributions_py",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_tests(
    name = "binomial_test",
    size = "small",
    srcs = ["python/kernel_tests/binomial_test.py"],
    additional_deps = [
        ":distributions_py",
        "//tensorflow/python:platform_test",
    ],
    tags = ["notsan"],
)

cuda_py_tests(
    name = "categorical_test",
    size = "small",
    srcs = ["python/kernel_tests/categorical_test.py"],
    additional_deps = [
        ":distributions_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_tests(
    name = "chi2_test",
    srcs = ["python/kernel_tests/chi2_test.py"],
    additional_deps = [
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
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
    tags = ["notsan"],
)

cuda_py_tests(
    name = "dirichlet_multinomial_test",
    size = "small",
    srcs = ["python/kernel_tests/dirichlet_multinomial_test.py"],
    additional_deps = [
        ":distributions_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
    tags = ["notsan"],
)

cuda_py_tests(
    name = "exponential_test",
    srcs = ["python/kernel_tests/exponential_test.py"],
    additional_deps = [
        ":distributions_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_tests(
    name = "gamma_test",
    srcs = ["python/kernel_tests/gamma_test.py"],
    additional_deps = [
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_tests(
    name = "inverse_gamma_test",
    srcs = ["python/kernel_tests/inverse_gamma_test.py"],
    additional_deps = [
        ":distributions_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_tests(
    name = "laplace_test",
    srcs = ["python/kernel_tests/laplace_test.py"],
    additional_deps = [
        ":distributions_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_tests(
    name = "multinomial_test",
    srcs = ["python/kernel_tests/multinomial_test.py"],
    additional_deps = [
        ":distributions_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
    tags = ["notsan"],
)

cuda_py_tests(
    name = "mvn_test",
    size = "small",
    srcs = ["python/kernel_tests/mvn_test.py"],
    additional_deps = [
        ":distributions_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_tests(
    name = "normal_test",
    size = "small",
    srcs = ["python/kernel_tests/normal_test.py"],
    additional_deps = [
        ":distributions_py",
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
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
    tags = ["notsan"],
)

cuda_py_tests(
    name = "uniform_test",
    size = "small",
    srcs = ["python/kernel_tests/uniform_test.py"],
    additional_deps = [
        ":distributions_py",
        "//tensorflow/python:framework_test_lib",
    ],
)

cuda_py_tests(
    name = "kullback_leibler_test",
    size = "small",
    srcs = ["python/kernel_tests/kullback_leibler_test.py"],
    additional_deps = [
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_tests(
    name = "normal_conjugate_posteriors_test",
    size = "small",
    srcs = ["python/kernel_tests/normal_conjugate_posteriors_test.py"],
    additional_deps = [
        ":distributions_py",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_tests(
    name = "transformed_distribution_test",
    size = "small",
    srcs = ["python/kernel_tests/transformed_distribution_test.py"],
    additional_deps = [
        ":distributions_py",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_tests(
    name = "shape_test",
    size = "small",
    srcs = ["python/kernel_tests/shape_test.py"],
    additional_deps = [
        ":distributions_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_tests(
    name = "bijector_test",
    size = "small",
    srcs = ["python/kernel_tests/bijector_test.py"],
    additional_deps = [
        ":distributions_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
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

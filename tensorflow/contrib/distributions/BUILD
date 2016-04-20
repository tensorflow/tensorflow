# Description:
#   Contains ops for statistical distributions (with pdf, cdf, sample, etc...).
#   APIs here are meant to evolve over time.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

load("//tensorflow:tensorflow.bzl", "cuda_py_tests")

py_library(
    name = "distributions_py",
    srcs = ["__init__.py"] + glob(["python/ops/*.py"]),
    srcs_version = "PY2AND3",
)

cuda_py_tests(
    name = "dirichlet_multinomial_test",
    srcs = ["python/kernel_tests/dirichlet_multinomial_test.py"],
    additional_deps = [
        ":distributions_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_tests(
    name = "gaussian_test",
    size = "small",
    srcs = ["python/kernel_tests/gaussian_test.py"],
    additional_deps = [
        ":distributions_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_tests(
    name = "gaussian_conjugate_posteriors_test",
    size = "small",
    srcs = ["python/kernel_tests/gaussian_conjugate_posteriors_test.py"],
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

# Description:
#   Contains ops for statistical distributions (with pdf, cdf, sample, etc...).
#   APIs here are meant to evolve over time.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

# load("//tensorflow:tensorflow.bzl", "cuda_py_tests")

py_library(
    name = "bayesflow_py",
    srcs = ["__init__.py"] + glob(["python/ops/*.py"]),
    srcs_version = "PY2AND3",
)

# cuda_py_tests(
#     name = "some_bayesflow_test",
#     size = "small",
#     srcs = ["python/kernel_tests/some_bayesflow_test.py"],
#     additional_deps = [
#         ":bayesflow_py",
#         "//tensorflow/python:framework_test_lib",
#         "//tensorflow/python:platform_test",
#     ],
# )

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

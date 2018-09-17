package(
    default_visibility = [
        "//tensorflow:internal",
    ],
)

licenses(["notice"])  # Apache 2.0

load("//tensorflow/python/tools/api/generator:api_gen.bzl", "gen_api_init_files")
load("//tensorflow/python/tools/api/generator:api_gen.bzl", "ESTIMATOR_API_INIT_FILES")

gen_api_init_files(
    name = "estimator_python_api_gen",
    api_name = "estimator",
    output_files = ESTIMATOR_API_INIT_FILES,
    output_package = "tensorflow.python.estimator.api",
    package = "tensorflow.python.estimator",
    package_dep = "//tensorflow/python/estimator:estimator_py",
)

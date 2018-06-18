package(
    default_visibility = [
        "//tensorflow:internal",
    ],
)

licenses(["notice"])  # Apache 2.0

load("//tensorflow/tools/api/generator:api_gen.bzl", "gen_api_init_files")
load("//tensorflow/tools/api/generator:api_gen.bzl", "ESTIMATOR_API_INIT_FILES")

gen_api_init_files(
    name = "estimator_python_api_gen",
    api_name = "estimator",
    output_files = ESTIMATOR_API_INIT_FILES,
    package = "tensorflow.python.estimator",
)

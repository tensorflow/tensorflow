# Description:
# Scripts used to generate TensorFlow Python API.

load("//tensorflow:tensorflow.bzl", "py_test")
load("//tensorflow/python/tools/api/generator:api_init_files.bzl", "KERAS_API_INIT_FILES", "TENSORFLOW_API_INIT_FILES")
load("//tensorflow/python/tools/api/generator:api_init_files_v1.bzl", "KERAS_API_INIT_FILES_V1", "TENSORFLOW_API_INIT_FILES_V1")
load("//tensorflow/python/tools/api/generator:api_gen.bzl", "TENSORFLOW_API_GEN_PACKAGES")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

exports_files(
    [
        "create_python_api.py",
    ],
)

py_library(
    name = "create_python_api",
    srcs = ["//tensorflow/python/tools/api/generator:create_python_api.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/python:util",
        "//tensorflow/python/tools/api/generator:doc_srcs",
        "//tensorflow/python/util:fast_module_type",
    ],
)

py_library(
    name = "doc_srcs",
    srcs = ["doc_srcs.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/python:util",
    ],
)

py_test(
    name = "create_python_api_test",
    srcs = [
        "create_python_api.py",
        "create_python_api_test.py",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":doc_srcs",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:no_contrib",
    ],
)

py_test(
    name = "tensorflow_doc_srcs_test",
    srcs = ["doc_srcs_test.py"],
    args = [
        "--package=tensorflow.python",
        "--api_name=tensorflow",
    ] + KERAS_API_INIT_FILES + KERAS_API_INIT_FILES_V1 + TENSORFLOW_API_INIT_FILES + TENSORFLOW_API_INIT_FILES_V1,
    main = "doc_srcs_test.py",
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":doc_srcs",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:no_contrib",
    ],
)

py_test(
    name = "output_init_files_test",
    srcs = ["output_init_files_test.py"],
    args = [
        "--packages=" + ",".join(TENSORFLOW_API_GEN_PACKAGES),
    ],
    data = [
        "api_init_files.bzl",
        "api_init_files_v1.bzl",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "no_pip",
    ],
    deps = [
        "//tensorflow/lite/python:analyzer",
        "//tensorflow/lite/python:lite",
        "//tensorflow/lite/python/authoring",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_combinations",
        "//tensorflow/python:modules_with_exports",
        "//tensorflow/python:no_contrib",
    ],
)

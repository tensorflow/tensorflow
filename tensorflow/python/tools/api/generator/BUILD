# Description:
# Scripts used to generate TensorFlow Python API.

load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load("//tensorflow:strict.default.bzl", "py_strict_library", "py_strict_test")
load("//tensorflow/python/tools/api/generator:api_gen.bzl", "TENSORFLOW_API_GEN_PACKAGES")
load("//tensorflow/python/tools/api/generator:api_init_files.bzl", "KERAS_API_INIT_FILES", "TENSORFLOW_API_INIT_FILES")
load("//tensorflow/python/tools/api/generator:api_init_files_v1.bzl", "KERAS_API_INIT_FILES_V1", "TENSORFLOW_API_INIT_FILES_V1")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

exports_files(
    [
        "create_python_api.py",
    ],
)

py_strict_library(
    name = "create_python_api",
    srcs = ["create_python_api.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":doc_srcs",
        "//tensorflow/python/util:tf_decorator_py",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "doc_srcs",
    srcs = ["doc_srcs.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_test(
    name = "create_python_api_test",
    srcs = ["create_python_api_test.py"],
    deps = [
        ":create_python_api",
        # copybara:uncomment "//third_party/py/google/protobuf:use_fast_cpp_protos",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/util:tf_decorator_py",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_test(
    name = "tensorflow_doc_srcs_test",
    srcs = ["doc_srcs_test.py"],
    args = [
        "--package=tensorflow.python",
        "--api_name=tensorflow",
    ] + KERAS_API_INIT_FILES + KERAS_API_INIT_FILES_V1 + TENSORFLOW_API_INIT_FILES + TENSORFLOW_API_INIT_FILES_V1,
    main = "doc_srcs_test.py",
    deps = [
        ":doc_srcs",
        # copybara:uncomment "//third_party/py/google/protobuf:use_fast_cpp_protos",
        "//tensorflow/python:no_contrib",
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_strict_test(
    name = "output_init_files_test",
    srcs = ["output_init_files_test.py"],
    args = [
        "--packages=" + ",".join(TENSORFLOW_API_GEN_PACKAGES),
    ],
    data = [
        "api_init_files.bzl",
        "api_init_files_v1.bzl",
    ],
    tags = [
        "no_pip",
    ],
    deps = [
        # copybara:uncomment "//third_party/py/google/protobuf:use_fast_cpp_protos",
        "//tensorflow/dtensor/python:dtensor",
        "//tensorflow/lite/python:analyzer",
        "//tensorflow/lite/python:lite",
        "//tensorflow/lite/python/authoring",
        "//tensorflow/python:modules_with_exports",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:resource_loader",
        "//tensorflow/python/util:tf_decorator_py",
    ],
)

bzl_library(
    name = "api_init_files",
    srcs = ["api_init_files.bzl"],
    visibility = ["//tensorflow/python/tools/api/generator2:__subpackages__"],
)

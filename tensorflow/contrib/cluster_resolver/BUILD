# Description: Operations defined for Cluster Resolvers

load("//tensorflow:tensorflow.bzl", "tf_py_test")

package(
    default_visibility = [
        "//tensorflow:__subpackages__",
    ],
)

licenses(["notice"])  # Apache 2.0

py_library(
    name = "cluster_resolver_pip",
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":cluster_resolver_py",
    ],
)

py_library(
    name = "cluster_resolver_py",
    srcs = glob([
        "__init__.py",
        "python/training/*.py",
    ]),
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = ["//tensorflow/python/distribute/cluster_resolver:cluster_resolver_lib"],
)

tf_py_test(
    name = "cluster_resolver_initialization_test",
    srcs = ["cluster_resolver_initialization_test.py"],
    additional_deps = [
        ":cluster_resolver_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:training",
    ],
    main = "cluster_resolver_initialization_test.py",
)

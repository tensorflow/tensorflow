# Description: Operations defined for Cluster Resolvers

load("//tensorflow:tensorflow.bzl", "tf_py_test")

package(
    default_visibility = [
        "//tensorflow:__subpackages__",
    ],
)

licenses(["notice"])  # Apache 2.0

filegroup(
    name = "all_files",
    srcs = glob(
        include = [
            "**/*",
        ],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
)

py_library(
    name = "cluster_resolver_pip",
    srcs = [
        "__init__.py",
        "python/training/__init__.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":cluster_resolver_py",
        ":gce_cluster_resolver_py",
        ":tpu_cluster_resolver_py",
        "//tensorflow/python:util",
    ],
)

py_library(
    name = "cluster_resolver_py",
    srcs = ["python/training/cluster_resolver.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:training",
    ],
)

py_library(
    name = "gce_cluster_resolver_py",
    srcs = ["python/training/gce_cluster_resolver.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":cluster_resolver_py",
        "//tensorflow/python:training",
    ],
)

py_library(
    name = "tpu_cluster_resolver_py",
    srcs = ["python/training/tpu_cluster_resolver.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":cluster_resolver_py",
        "//tensorflow/python:training",
    ],
)

tf_py_test(
    name = "cluster_resolver_py_test",
    srcs = ["python/training/cluster_resolver_test.py"],
    additional_deps = [
        ":cluster_resolver_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:training",
    ],
    main = "python/training/cluster_resolver_test.py",
)

tf_py_test(
    name = "gce_cluster_resolver_py_test",
    size = "small",
    srcs = ["python/training/gce_cluster_resolver_test.py"],
    additional_deps = [
        ":cluster_resolver_py",
        ":gce_cluster_resolver_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:training",
    ],
    main = "python/training/gce_cluster_resolver_test.py",
)

tf_py_test(
    name = "tpu_cluster_resolver_py_test",
    size = "small",
    srcs = ["python/training/tpu_cluster_resolver_test.py"],
    additional_deps = [
        ":tpu_cluster_resolver_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:training",
    ],
    main = "python/training/tpu_cluster_resolver_test.py",
)

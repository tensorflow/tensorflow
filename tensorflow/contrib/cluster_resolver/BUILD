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
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
)

py_library(
    name = "cluster_resolver_py",
    srcs = [
        "python/training/__init__.py",
        "python/training/cluster_resolver.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:framework",
    ],
)

tf_py_test(
    name = "cluster_resolver_py_test",
    size = "small",
    srcs = ["python/training/cluster_resolver_test.py"],
    additional_deps = [
        ":cluster_resolver_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
    main = "python/training/cluster_resolver_test.py",
)

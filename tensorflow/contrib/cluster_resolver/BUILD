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
    srcs = [
        "__init__.py",
        "python/training/__init__.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":base_cluster_resolver_py",
        ":gce_cluster_resolver_py",
        ":kubernetes_cluster_resolver_py",
        ":slurm_cluster_resolver_py",
        ":tfconfig_cluster_resolver_py",
        ":tpu_cluster_resolver_py",
        "//tensorflow/python:util",
    ],
)

py_library(
    name = "base_cluster_resolver_py",
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
        ":base_cluster_resolver_py",
        "//tensorflow/python:training",
    ],
)

py_library(
    name = "tfconfig_cluster_resolver_py",
    srcs = ["python/training/tfconfig_cluster_resolver.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":base_cluster_resolver_py",
        "//tensorflow/python:training",
    ],
)

py_library(
    name = "tpu_cluster_resolver_py",
    srcs = ["python/training/tpu_cluster_resolver.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":base_cluster_resolver_py",
        "//tensorflow/python:training",
    ],
)

py_library(
    name = "slurm_cluster_resolver_py",
    srcs = ["python/training/slurm_cluster_resolver.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":base_cluster_resolver_py",
        "//tensorflow/python:training",
    ],
)

py_library(
    name = "kubernetes_cluster_resolver_py",
    srcs = ["python/training/kubernetes_cluster_resolver.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":base_cluster_resolver_py",
        "//tensorflow/python:training",
    ],
)

tf_py_test(
    name = "base_cluster_resolver_py_test",
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
    name = "tfconfig_cluster_resolver_py_test",
    size = "small",
    srcs = ["python/training/tfconfig_cluster_resolver_test.py"],
    additional_deps = [
        ":tfconfig_cluster_resolver_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:training",
    ],
    grpc_enabled = True,
    main = "python/training/tfconfig_cluster_resolver_test.py",
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
    grpc_enabled = True,
    main = "python/training/tpu_cluster_resolver_test.py",
)

tf_py_test(
    name = "slurm_cluster_resolver_py_test",
    size = "small",
    srcs = ["python/training/slurm_cluster_resolver_test.py"],
    additional_deps = [
        ":cluster_resolver_py",
        ":slurm_cluster_resolver_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:training",
    ],
    main = "python/training/slurm_cluster_resolver_test.py",
    tags = [],
)

tf_py_test(
    name = "kubernetes_cluster_resolver_py_test",
    size = "small",
    srcs = ["python/training/kubernetes_cluster_resolver_test.py"],
    additional_deps = [
        ":cluster_resolver_py",
        ":kubernetes_cluster_resolver_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:training",
    ],
    main = "python/training/kubernetes_cluster_resolver_test.py",
)

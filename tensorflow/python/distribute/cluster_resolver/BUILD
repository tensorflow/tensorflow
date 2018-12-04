# Description: Operations defined for Cluster Resolvers

load("//tensorflow:tensorflow.bzl", "tf_py_test")

package(
    default_visibility = [
        "//tensorflow:__subpackages__",
    ],
)

licenses(["notice"])  # Apache 2.0

py_library(
    name = "cluster_resolver_lib",
    srcs = [
        "__init__.py",
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
    srcs = ["cluster_resolver.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:training_server_lib",
    ],
)

py_library(
    name = "gce_cluster_resolver_py",
    srcs = ["gce_cluster_resolver.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":base_cluster_resolver_py",
        "//tensorflow/python:training_server_lib",
    ],
)

py_library(
    name = "tfconfig_cluster_resolver_py",
    srcs = ["tfconfig_cluster_resolver.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":base_cluster_resolver_py",
        "//tensorflow/python:training_server_lib",
    ],
)

py_library(
    name = "tpu_cluster_resolver_py",
    srcs = ["tpu_cluster_resolver.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":base_cluster_resolver_py",
        "//tensorflow/python:training_server_lib",
    ],
)

py_library(
    name = "slurm_cluster_resolver_py",
    srcs = ["slurm_cluster_resolver.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":base_cluster_resolver_py",
        "//tensorflow/python:training_server_lib",
    ],
)

py_library(
    name = "kubernetes_cluster_resolver_py",
    srcs = ["kubernetes_cluster_resolver.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":base_cluster_resolver_py",
        "//tensorflow/python:training_server_lib",
    ],
)

tf_py_test(
    name = "base_cluster_resolver_py_test",
    srcs = ["cluster_resolver_test.py"],
    additional_deps = [
        ":base_cluster_resolver_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:training_server_lib",
    ],
    main = "cluster_resolver_test.py",
)

tf_py_test(
    name = "gce_cluster_resolver_py_test",
    size = "small",
    srcs = ["gce_cluster_resolver_test.py"],
    additional_deps = [
        ":gce_cluster_resolver_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:training_server_lib",
    ],
    main = "gce_cluster_resolver_test.py",
)

tf_py_test(
    name = "tfconfig_cluster_resolver_py_test",
    size = "small",
    srcs = ["tfconfig_cluster_resolver_test.py"],
    additional_deps = [
        ":tfconfig_cluster_resolver_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:training_server_lib",
    ],
    grpc_enabled = True,
    main = "tfconfig_cluster_resolver_test.py",
)

tf_py_test(
    name = "tpu_cluster_resolver_py_test",
    size = "small",
    srcs = ["tpu_cluster_resolver_test.py"],
    additional_deps = [
        ":tpu_cluster_resolver_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:training_server_lib",
    ],
    grpc_enabled = True,
    main = "tpu_cluster_resolver_test.py",
)

tf_py_test(
    name = "slurm_cluster_resolver_py_test",
    size = "small",
    srcs = ["slurm_cluster_resolver_test.py"],
    additional_deps = [
        ":slurm_cluster_resolver_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:training_server_lib",
    ],
    main = "slurm_cluster_resolver_test.py",
    tags = [],
)

tf_py_test(
    name = "kubernetes_cluster_resolver_py_test",
    size = "small",
    srcs = ["kubernetes_cluster_resolver_test.py"],
    additional_deps = [
        ":kubernetes_cluster_resolver_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:training_server_lib",
    ],
    main = "kubernetes_cluster_resolver_test.py",
)

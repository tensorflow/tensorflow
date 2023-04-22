# Description: OSS only cluster resolvers

load("//tensorflow:tensorflow.bzl", "tf_py_test")
load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_additional_rpc_deps",
)

package(
    default_visibility = [
        "//tensorflow:internal",
    ],
    licenses = ["notice"],
)

py_library(
    name = "tpu_cluster_resolver_py",
    srcs = ["tpu_cluster_resolver.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:training_server_lib",
        "//tensorflow/python/distribute/cluster_resolver:base_cluster_resolver_py",
        "//tensorflow/python/tpu:tpu_system_metadata",
        "//tensorflow/python/tpu/client",
    ] + tf_additional_rpc_deps(),
)

tf_py_test(
    name = "tpu_cluster_resolver_py_test",
    size = "small",
    srcs = ["tpu_cluster_resolver_test.py"],
    grpc_enabled = True,
    main = "tpu_cluster_resolver_test.py",
    python_version = "PY3",
    deps = [
        ":tpu_cluster_resolver_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:training_server_lib",
        "//tensorflow/python/tpu/client",
    ],
)

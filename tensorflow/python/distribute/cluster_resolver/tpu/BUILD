# Description: OSS only cluster resolvers
load("//tensorflow:strict.default.bzl", "py_strict_library")
load("//tensorflow:tensorflow.default.bzl", "tf_py_strict_test")
load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_additional_rpc_deps",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow:internal",
    ],
    licenses = ["notice"],
)

py_strict_library(
    name = "tpu_cluster_resolver_py",
    srcs = ["tpu_cluster_resolver.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/core/protobuf/tpu:topology_proto_py",
        "//tensorflow/python/distribute/cluster_resolver:base_cluster_resolver_py",
        "//tensorflow/python/eager:remote",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/tpu:tpu_strategy_util",
        "//tensorflow/python/tpu:tpu_system_metadata",
        "//tensorflow/python/tpu/client",
        "//tensorflow/python/training:server_lib",
        "//tensorflow/python/util:compat",
    ] + tf_additional_rpc_deps(),
)

tf_py_strict_test(
    name = "tpu_cluster_resolver_py_test",
    size = "small",
    srcs = ["tpu_cluster_resolver_test.py"],
    grpc_enabled = True,
    main = "tpu_cluster_resolver_test.py",
    python_version = "PY3",
    deps = [
        ":tpu_cluster_resolver_py",
        "//tensorflow/core/protobuf/tpu:topology_proto_py",
        "//tensorflow/python/client:session",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/tpu/client",
        "//tensorflow/python/training:server_lib",
        "//tensorflow/python/util:compat",
    ],
)

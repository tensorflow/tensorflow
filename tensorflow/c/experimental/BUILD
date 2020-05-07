# Description:
# Experimental C APIs for TensorFlow.

load(
    "//tensorflow:tensorflow.bzl",
    "tf_copts",
    "tf_cuda_library",
)
load("//tensorflow:tensorflow.bzl", "tf_cuda_cc_test")

package(
    licenses = ["notice"],  # Apache 2.0
)

tf_cuda_library(
    name = "rendezvous_internal",
    srcs = [
        "rendezvous.cc",
    ],
    hdrs = [
        "rendezvous.h",
        "rendezvous_internal.h",
    ],
    copts = tf_copts(),
    visibility = ["//tensorflow/c:__subpackages__"],
    deps = [
        "//tensorflow/c:c_api_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/distributed_runtime:base_rendezvous_mgr",
        "//tensorflow/core/distributed_runtime:worker_env",
        "//tensorflow/core/distributed_runtime/rpc:grpc_server_lib",
    ],
)

tf_cuda_library(
    name = "rendezvous",
    hdrs = [
        "rendezvous.h",
    ],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":rendezvous_internal",
        "//tensorflow/c:c_api",
    ],
)

tf_cuda_library(
    name = "network_internal",
    srcs = [
        "network.cc",
    ],
    hdrs = [
        "network.h",
        "network_internal.h",
    ],
    copts = tf_copts(),
    visibility = ["//tensorflow/c:__subpackages__"],
    deps = [
        ":rendezvous_internal",
        "//tensorflow/c:c_api_internal",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/distributed_runtime:server_lib",
        "//tensorflow/core/distributed_runtime:worker_env",
        "//tensorflow/core/distributed_runtime/rpc:grpc_server_lib",
    ],
)

tf_cuda_library(
    name = "network",
    hdrs = [
        "network.h",
    ],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":network_internal",
        ":rendezvous",
        "//tensorflow/c:c_api",
    ],
)

# -----------------------------------------------------------------------------
# Tests

tf_cuda_cc_test(
    name = "network_test",
    size = "medium",
    srcs = ["network_test.cc"],
    tags = ["noasan"],
    # We must ensure that the dependencies can be dynamically linked since
    # the shared library must be able to use core:framework.
    # linkstatic = tf_kernel_tests_linkstatic(),
    deps = [
        ":network",
        ":network_internal",
        ":rendezvous",
        ":rendezvous_internal",
        "//tensorflow/c:c_api",
        "//tensorflow/c:env",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/distributed_runtime:rendezvous_mgr_interface",
        "//tensorflow/core/distributed_runtime:server_lib",
        "//tensorflow/core/distributed_runtime:session_mgr",
        "//tensorflow/core/distributed_runtime:worker_env",
        "//tensorflow/core/distributed_runtime:worker_session",
        "//tensorflow/core/distributed_runtime/rpc:async_service_interface",
        "//tensorflow/core/distributed_runtime/rpc:grpc_server_lib",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

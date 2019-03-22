# Description: Operations defined for XRT

licenses(["notice"])  # Apache 2.0

package(
    default_visibility = [
        "//learning/brain:__subpackages__",
        "//tensorflow/compiler/xrt:__subpackages__",
    ],
)

load("//tensorflow:tensorflow.bzl", "tf_cc_binary", "tf_cc_test")

cc_library(
    name = "xrt_grpc_eager_client",
    srcs = ["xrt_grpc_eager_client.cc"],
    hdrs = ["xrt_grpc_eager_client.h"],
    deps = [
        "//tensorflow:grpc++",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/core:eager_service_proto_cc",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:worker_proto_cc",
        "//tensorflow/core/distributed_runtime/rpc:grpc_channel",
        "//tensorflow/core/distributed_runtime/rpc:grpc_client_cq_tag",
        "//tensorflow/core/distributed_runtime/rpc:grpc_state",
        "//tensorflow/core/distributed_runtime/rpc:grpc_util",
        "@com_google_absl//absl/synchronization",
    ],
)

tf_cc_test(
    name = "xrt_client_test",
    srcs = ["xrt_client_test.cc"],
    data = [":xrt_testlib_server"],
    deps = [
        ":xrt_grpc_eager_client",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/core:eager_service_proto_cc",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/distributed_runtime/rpc:grpc_channel",
        "//tensorflow/core/distributed_runtime/rpc:grpc_server_lib",
        "//tensorflow/core/distributed_runtime/rpc:grpc_session",
        "//tensorflow/core/distributed_runtime/rpc:grpc_testlib",
    ],
)

tf_cc_binary(
    name = "xrt_testlib_server",
    testonly = 1,
    deps = [
        "//tensorflow/compiler/jit:xla_cpu_device",
        "//tensorflow/compiler/xrt:xrt_server",
        "//tensorflow/core/distributed_runtime/rpc:grpc_testlib_server_main",
    ],
)

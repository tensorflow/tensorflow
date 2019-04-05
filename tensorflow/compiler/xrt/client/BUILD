# Description: Operations defined for XRT

licenses(["notice"])  # Apache 2.0

package(
    default_visibility = [
        "//tensorflow:internal",
        "//tensorflow/compiler:__subpackages__",
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
        "//tensorflow/core/distributed_runtime:call_options",
        "//tensorflow/core/distributed_runtime/rpc:grpc_channel",
        "//tensorflow/core/distributed_runtime/rpc:grpc_client_cq_tag",
        "//tensorflow/core/distributed_runtime/rpc:grpc_state",
        "//tensorflow/core/distributed_runtime/rpc:grpc_util",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "xrt_tf_client",
    srcs = ["xrt_tf_client.cc"],
    hdrs = ["xrt_tf_client.h"],
    deps = [
        ":xrt_grpc_eager_client",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/core:eager_service_proto_cc",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/distributed_runtime:call_options",
        "//tensorflow/core/distributed_runtime:request_id",
        "//tensorflow/core/distributed_runtime/rpc:grpc_channel",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "xrt_client",
    srcs = ["xrt_client.cc"],
    hdrs = ["xrt_client.h"],
    deps = [
        ":xrt_tf_client",
        "//tensorflow/compiler/xla:array2d",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/service:computation_placer",
        "//tensorflow/compiler/xla/service:hlo_proto",
        "//tensorflow/compiler/xrt:xrt_proto",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/protobuf/tpu:topology_proto_cc",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

tf_cc_test(
    name = "xrt_client_test",
    srcs = ["xrt_client_test.cc"],
    data = [":xrt_testlib_server"],
    deps = [
        ":xrt_client",
        ":xrt_grpc_eager_client",
        ":xrt_tf_client",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla/client:xla_builder",
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

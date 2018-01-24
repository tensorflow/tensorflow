# Description:
#   GPU Direct RDMA Out-of-Band Tensor transport for TensorFlow.

package(default_visibility = [
    "//tensorflow:__subpackages__",
])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)

filegroup(
    name = "c_srcs",
    data = glob([
        "**/*.cc",
        "**/*.h",
    ]),
)

load(
    "//tensorflow:tensorflow.bzl",
    "tf_cuda_library",
)

# For platform specific build config
load(
    "//tensorflow/core:platform/default/build_config.bzl",
    "tf_proto_library_cc",
)

tf_proto_library_cc(
    name = "gdr_proto",
    srcs = ["gdr.proto"],
    cc_api_version = 2,
    visibility = [
        "//tensorflow:__subpackages__",
    ],
)

tf_cuda_library(
    name = "gdr_memory_manager",
    srcs = ["gdr_memory_manager.cc"],
    hdrs = ["gdr_memory_manager.h"],
    linkopts = select({
        "//tensorflow:with_gdr_support": [
            "-libverbs",
            "-lrdmacm",
        ],
        "//conditions:default": [],
    }),
    deps = [
        ":gdr_proto_cc",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:gpu_runtime",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_cuda_library(
    name = "gdr_worker",
    srcs = ["gdr_worker.cc"],
    hdrs = ["gdr_worker.h"],
    deps = [
        ":gdr_memory_manager",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:gpu_runtime",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/distributed_runtime:graph_mgr",
        "//tensorflow/core/distributed_runtime:recent_request_ids",
        "//tensorflow/core/distributed_runtime:rendezvous_mgr_interface",
        "//tensorflow/core/distributed_runtime:worker",
        "//tensorflow/core/distributed_runtime:worker_cache",
        "//tensorflow/core/distributed_runtime:worker_session",
        "//tensorflow/core/distributed_runtime/rpc:grpc_call",
        "//tensorflow/core/distributed_runtime/rpc:grpc_tensor_coding",
        "//tensorflow/core/distributed_runtime/rpc:grpc_util",
        "//tensorflow/core/distributed_runtime/rpc:grpc_worker_service",
    ],
)

cc_library(
    name = "gdr_rendezvous_mgr",
    srcs = ["gdr_rendezvous_mgr.cc"],
    hdrs = ["gdr_rendezvous_mgr.h"],
    deps = [
        ":gdr_memory_manager",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/distributed_runtime:base_rendezvous_mgr",
        "//tensorflow/core/distributed_runtime:request_id",
        "//tensorflow/core/distributed_runtime:tensor_coding",
        "//tensorflow/core/distributed_runtime:worker_cache",
        "//tensorflow/core/distributed_runtime:worker_env",
        "//tensorflow/core/distributed_runtime:worker_interface",
    ],
)

cc_library(
    name = "gdr_server_lib",
    srcs = ["gdr_server_lib.cc"],
    hdrs = ["gdr_server_lib.h"],
    linkstatic = 1,  # Seems to be needed since alwayslink is broken in bazel
    deps = [
        ":gdr_memory_manager",
        ":gdr_rendezvous_mgr",
        ":gdr_worker",
        "//tensorflow/core/distributed_runtime/rpc:grpc_server_lib",
    ],
    alwayslink = 1,
)

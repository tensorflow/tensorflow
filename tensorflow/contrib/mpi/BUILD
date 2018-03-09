# Description:
#   MPI based communication interfaces and implementations for TensorFlow.

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

# For platform specific build config
load(
    "//tensorflow/core:platform/default/build_config.bzl",
    "tf_proto_library_cc",
)

tf_proto_library_cc(
    name = "mpi_msg_proto",
    srcs = ["mpi_msg.proto"],
    cc_api_version = 2,
    protodeps = ["//tensorflow/core:worker_proto"],
    visibility = [
        "//tensorflow:__subpackages__",
    ],
)

cc_library(
    name = "mpi_utils",
    srcs = ["mpi_utils.cc"],
    hdrs = ["mpi_utils.h"],
    deps = [
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//third_party/mpi",
    ],
)

cc_library(
    name = "mpi_rendezvous_mgr",
    srcs = ["mpi_rendezvous_mgr.cc"],
    hdrs = ["mpi_rendezvous_mgr.h"],
    deps = [
        ":mpi_msg_proto_cc",
        ":mpi_utils",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:gpu_runtime",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_cc",
        "//tensorflow/core:worker_proto_cc",
        "//tensorflow/core/distributed_runtime:base_rendezvous_mgr",
        "//tensorflow/core/distributed_runtime:recent_request_ids",
        "//tensorflow/core/distributed_runtime:request_id",
        "//tensorflow/core/distributed_runtime:session_mgr",
        "//tensorflow/core/distributed_runtime:tensor_coding",
        "//tensorflow/core/distributed_runtime:worker_env",
        "//third_party/mpi",
    ],
)

cc_library(
    name = "mpi_server_lib",
    srcs = ["mpi_server_lib.cc"],
    hdrs = ["mpi_server_lib.h"],
    linkstatic = 1,  # Seems to be needed since alwayslink is broken in bazel
    deps = [
        ":mpi_rendezvous_mgr",
        "//tensorflow/core/distributed_runtime/rpc:grpc_server_lib",
    ],
    alwayslink = 1,
)

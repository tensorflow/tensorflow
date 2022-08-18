# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "tf_kernel_library")
load("//tensorflow:tensorflow.bzl", "tf_gen_op_wrapper_py")
load("//tensorflow/distribute/experimental/rpc/kernels/oss:defs.bzl", "grpc_credentials_dependency")

package(
    default_visibility = [
        "//tensorflow:__subpackages__",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "grpc_rpc_service",
    srcs = ["grpc_rpc_service.h"],
    hdrs = ["grpc_rpc_service.h"],
    deps = [
        "//tensorflow/distribute/experimental/rpc/proto:tf_rpc_service_cc_grpc_proto",
        "//tensorflow/distribute/experimental/rpc/proto:tf_rpc_service_proto_cc",
        "//tensorflow/stream_executor/platform",
        "@com_github_grpc_grpc//:grpc++",
    ],
    alwayslink = 1,
)

cc_library(
    name = "grpc_credentials",
    hdrs = ["grpc_credentials.h"],
    deps = [
        "@com_github_grpc_grpc//:grpc++",
    ] + grpc_credentials_dependency(),
)

tf_kernel_library(
    name = "rpc_ops",
    prefix = "rpc_ops",
    deps = [
        ":grpc_credentials",
        ":grpc_rpc_service",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/data:dataset_utils",
        "//tensorflow/core/distributed_runtime/rpc:grpc_client_cq_tag",
        "//tensorflow/core/distributed_runtime/rpc:grpc_state",
        "//tensorflow/core/distributed_runtime/rpc:grpc_util",
        "//tensorflow/distribute/experimental/rpc/proto:tf_rpc_service_cc_grpc_proto",
        "//tensorflow/distribute/experimental/rpc/proto:tf_rpc_service_proto_cc",
        "//tensorflow/stream_executor/platform",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
    alwayslink = 1,
)

tf_gen_op_wrapper_py(
    name = "gen_rpc_ops",
    out = "gen_rpc_ops.py",
    deps = [
        ":rpc_ops",
    ],
)

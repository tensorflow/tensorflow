load("@local_xla//xla/tsl/platform:rules_cc.bzl", "cc_library")
load("//tensorflow:strict.default.bzl", "py_strict_library")
load("//tensorflow:tensorflow.bzl", "tf_gen_op_wrapper_py")
load("//tensorflow:tensorflow.default.bzl", "tf_kernel_library")
load("//tensorflow/distribute/experimental/rpc/kernels/oss:defs.bzl", "grpc_credentials_dependency")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
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

# Note: we cannot call the source file "rpc_ops_metadata.cc", as we
# would normally like to do, because tf_kernel_library below uses a
# glob (!) of sources with the pattern `prefix + "*.cc"`, which would
# cause the metadata translation unit to be compiled and linked twice.
cc_library(
    name = "rpc_ops_metadata",
    srcs = ["metadata_for_rpc_ops.cc"],
    deps = ["//tensorflow/core:framework"],
    alwayslink = 1,
)

tf_kernel_library(
    name = "rpc_ops",
    prefix = "rpc_ops",
    deps = [
        ":grpc_credentials",
        ":grpc_rpc_service",
        ":rpc_ops_metadata",
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
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
    alwayslink = 1,
)

tf_gen_op_wrapper_py(
    name = "gen_rpc_ops",
    out = "gen_rpc_ops.py",
    extra_py_deps = [
        "//tensorflow/python:pywrap_tfe",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ],
    py_lib_rule = py_strict_library,
    deps = [":rpc_ops_metadata"],
)

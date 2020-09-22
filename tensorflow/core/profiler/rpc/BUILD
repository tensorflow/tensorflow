load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//tensorflow:tensorflow.bzl", "tf_external_workspace_visible")  # buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "tf_grpc_cc_dependency")  # buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "tf_pybind_cc_library_wrapper")  # buildifier: disable=same-origin-load
load("//tensorflow/core/profiler/builds:build_config.bzl", "tf_profiler_alias")

package(
    default_visibility = [
        "//tensorflow/core/profiler:internal",
    ],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "grpc",
    hdrs = ["grpc.h"],
    deps = [
        tf_profiler_alias("//tensorflow/core/profiler/rpc/", "grpc"),
        tf_grpc_cc_dependency(),
    ],
)

exports_files(
    [
        "grpc.h",
    ],
    visibility = ["//tensorflow/core/profiler/rpc:__subpackages__"],
)

cc_library(
    name = "profiler_service_impl",
    srcs = ["profiler_service_impl.cc"],
    hdrs = ["profiler_service_impl.h"],
    features = ["-layering_check"],
    visibility = tf_external_workspace_visible(
        [
            "//tensorflow/core/data/service:__pkg__",
            "//tensorflow/core/distributed_runtime/rpc:__pkg__",
            "//tensorflow_serving/model_servers:__pkg__",
        ],
    ),
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core/profiler:profiler_service_proto_cc",
        "//tensorflow/core/profiler/lib:profiler_session_headers",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
        "@com_google_absl//absl/memory",
        tf_grpc_cc_dependency(),
    ],
)

cc_library(
    name = "profiler_server_impl",
    srcs = ["profiler_server.cc"],
    hdrs = ["profiler_server.h"],
    visibility = [
        "//tensorflow/compiler/xla/python:__pkg__",
        "//tensorflow/python:__pkg__",
        "//tensorflow/python/profiler/internal:__pkg__",
    ],
    deps = [
        ":profiler_service_impl",
        "//tensorflow/core:lib",
        "//tensorflow/core/profiler:profiler_service_proto_cc",
        "@com_google_absl//absl/strings",
        tf_grpc_cc_dependency(),
    ],
    alwayslink = True,
)

tf_pybind_cc_library_wrapper(
    name = "profiler_server_headers",
    visibility = [
        "//tensorflow/python/profiler/internal:__pkg__",
    ],
    deps = [":profiler_server_impl"],
)

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
    "tf_cc_test",
)
load("//tensorflow:tensorflow.bzl", "tf_cc_binary")

# For platform specific build config
load(
    "//tensorflow/core:platform/default/build_config.bzl",
    "tf_kernel_tests_linkstatic",
)

package(default_visibility = [
    "//tensorflow:internal",
])

COMMON_COPTS = [
    "-std=gnu++1y",
    "-DFMT_HEADER_ONLY",
    "-DNO_EXCEPTION_HACK",
    "-DNO_EXCEPTION_INTERCEPT",
    "-DHAVE_LZ4_COMPRESS_DEFAULT",
]

cc_library(
    name = "seastar_tensor_coding",
    srcs = [
        "seastar_tensor_coding.cc",
        "seastar_message.cc",
    ],
    hdrs = [
        "seastar_tensor_coding.h",
        "seastar_message.h",
    ],
    deps = [
        "//tensorflow/core:core_cpu",
    ],
)

cc_library(
    name = "seastar_worker_service",
    srcs = [
        "seastar_worker_service.cc",
        "seastar_client_tag.cc",
        "seastar_server_tag.cc",
        "seastar_tag_factory.cc",
    ],
    hdrs = [
        "seastar_worker_service_method.h",
        "seastar_worker_service.h",
        "seastar_worker_interface.h",
        "seastar_client_tag.h",
        "seastar_server_tag.h",
        "seastar_tag_factory.h",
    ],
    linkstatic = 1,
    copts = COMMON_COPTS,
    deps = [
        "@seastar//:seastar",
        ":seastar_tensor_coding",
        "//tensorflow/core:lib",
        "//tensorflow/core:master_proto_cc",
        "//tensorflow/core:worker_proto_cc",
        "//tensorflow/core/distributed_runtime:worker",
        "//tensorflow/core/distributed_runtime:worker_env",
        "//tensorflow/core/distributed_runtime:call_options",
        "//tensorflow/core:gpu_runtime",
    ],
    alwayslink = 1,
)

cc_library(
    name = "seastar_cpuset",
    srcs = ["seastar_cpuset.cc"],
    hdrs = ["seastar_cpuset.h"],
    linkstatic = 1,
    deps = ["//tensorflow/core:lib"],
    alwayslink = 1,
)

cc_library(
    name = "seastar_engine",
    srcs = [
        "seastar_engine.cc",
        "seastar_client.cc",
    ],
    hdrs = [
        "seastar_engine.h",
        "seastar_client.h",
    ],
    linkstatic = 1,
    copts = COMMON_COPTS,
    deps = [
        ":seastar_cpuset",
        ":seastar_worker_service",
    ],
    alwayslink = 1,
)

cc_library(
    name = "seastar_remote_worker",
    srcs = ["seastar_remote_worker.cc"],
    hdrs = ["seastar_remote_worker.h"],
    linkstatic = 1,
    copts = COMMON_COPTS,
    deps = [
        ":seastar_worker_service",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:lib",
        "//tensorflow/core:framework",
        "//tensorflow/core/distributed_runtime:worker_cache_logger",
    ],
    alwayslink = 1,
)

cc_library(
    name = "seastar_worker_cache",
    srcs = [
        "seastar_worker_cache.cc",
        "seastar_channel_cache.cc",
    ],
    hdrs = [
        "seastar_worker_cache.h",
        "seastar_channel_cache.h",
    ],
    linkstatic = 1,
    copts = COMMON_COPTS,
    deps = [
        ":seastar_remote_worker",
        ":seastar_engine",
        "//tensorflow/core/distributed_runtime:worker_cache_partial",
    ],
    alwayslink = 1,
)

cc_library(
    name = "seastar_rendezvous_mgr",
    srcs = ["seastar_rendezvous_mgr.cc"],
    hdrs = ["seastar_rendezvous_mgr.h"],
    linkstatic = 1,
    copts = COMMON_COPTS,
    deps = [
        ":seastar_worker_cache",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/distributed_runtime:base_rendezvous_mgr",
        "//tensorflow/core/distributed_runtime:worker_cache",
        "//tensorflow/core/distributed_runtime:worker_env",
        "//tensorflow/core/distributed_runtime:worker_interface",
    ],
    alwayslink = 1,
)

cc_library(
    name = "seastar_server_lib",
    srcs = ["seastar_server_lib.cc"],
    hdrs = ["seastar_server_lib.h"],
    linkstatic = 1,
    copts = COMMON_COPTS,
    deps = [
        ":seastar_rendezvous_mgr",
        "//tensorflow/core/distributed_runtime:local_master",
        "//tensorflow/core/distributed_runtime:master",
        "//tensorflow/core/distributed_runtime:master_env",
        "//tensorflow/core/distributed_runtime:master_session",
        "//tensorflow/core/distributed_runtime:server_lib",
        "//tensorflow/core/distributed_runtime/rpc:grpc_master_service",
        "//tensorflow/core/distributed_runtime/rpc:grpc_server_lib",
    ],
    alwayslink = 1,
)

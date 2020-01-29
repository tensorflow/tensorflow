# Experimental extensions to the C API for eager execution of kernels.

load(
    "//tensorflow:tensorflow.bzl",
    "tf_copts",
    "tf_cuda_cc_test",
    "tf_cuda_library",
    "tfe_xla_copts",
)
load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_kernel_tests_linkstatic",
)
load(
    "//tensorflow/core/platform:build_config_root.bzl",
    "tf_cuda_tests_tags",
)

package(
    licenses = ["notice"],  # Apache 2.0
)

tf_cuda_library(
    name = "c_api",
    srcs = [
        "c_api.cc",
        "c_api_debug.cc",
        "c_api_experimental.h",
        "c_api_internal.h",
        "tensor_handle_interface.h",
    ],
    hdrs = ["c_api.h"],
    copts = tf_copts() + tfe_xla_copts(),
    visibility = ["//visibility:public"],
    deps = select({
        "//tensorflow:android": [
            "//tensorflow/core:android_tensorflow_lib_lite",
        ],
        "//conditions:default": [
            "@com_google_absl//absl/algorithm:container",
            "@com_google_absl//absl/container:fixed_array",
            "//tensorflow/c:c_api",
            "//tensorflow/c:c_api_internal",
            "//tensorflow/c:tf_tensor_internal",
            "//tensorflow/core:core_cpu",
            "//tensorflow/core/common_runtime/eager:attr_builder",
            "//tensorflow/core/common_runtime/eager:context",
            "//tensorflow/core/common_runtime/eager:eager_executor",
            "//tensorflow/core/common_runtime/eager:execute",
            "//tensorflow/core/common_runtime/eager:kernel_and_device",
            "//tensorflow/core/common_runtime/eager:tensor_handle",
            "//tensorflow/core/common_runtime/eager:copy_to_device_node",
            "//tensorflow/core:core_cpu_internal",
            "//tensorflow/core:framework",
            "//tensorflow/core:framework_internal",
            "//tensorflow/core:lib",
            "//tensorflow/core:lib_internal",
            "//tensorflow/core/platform:errors",
            "//tensorflow/core:protos_all_cc",
            "//tensorflow/core/profiler/lib:traceme",
        ],
    }) + select({
        "//tensorflow:with_xla_support": [
            "//tensorflow/compiler/tf2xla:xla_compiler",
            "//tensorflow/compiler/jit",
            "//tensorflow/compiler/jit:xla_device",
        ],
        "//conditions:default": [],
    }) + [
        "@com_google_absl//absl/memory",
        "//tensorflow/core/common_runtime/eager:eager_operation",
        "//tensorflow/core/distributed_runtime/eager:remote_mgr",
        "//tensorflow/core/distributed_runtime/eager:cluster_function_library_runtime",
        "//tensorflow/core/distributed_runtime/eager:eager_client",
        "//tensorflow/core/distributed_runtime/rpc/eager:grpc_eager_client",
        "//tensorflow/core/distributed_runtime/rpc:grpc_channel",
        "//tensorflow/core/distributed_runtime/rpc:grpc_server_lib",
        "//tensorflow/core/distributed_runtime/rpc:grpc_worker_cache",
        "//tensorflow/core/distributed_runtime/rpc:grpc_worker_service",
        "//tensorflow/core/distributed_runtime/rpc:rpc_rendezvous_mgr",
        "//tensorflow/core/distributed_runtime:remote_device",
        "//tensorflow/core/distributed_runtime:server_lib",
        "//tensorflow/core/distributed_runtime:worker_env",
        "//tensorflow/core/profiler/lib:profiler_lib",
        "//tensorflow/core/profiler/lib:profiler_session",
        "//tensorflow/core:gpu_runtime",
    ],
    alwayslink = 1,
)

filegroup(
    name = "pywrap_required_hdrs",
    srcs = [
        "c_api_experimental.h",
        "c_api_internal.h",
        "tensor_handle_interface.h",
    ],
    visibility = [
        "//tensorflow/core:__pkg__",
        "//tensorflow/python:__pkg__",
    ],
)

tf_cuda_library(
    name = "c_api_internal",
    srcs = [
        "c_api_experimental.h",
        "tensor_handle_interface.h",
    ],
    hdrs = ["c_api_internal.h"],
    visibility = [
        "//learning/deepmind/courier:__subpackages__",
        "//tensorflow:internal",
    ],
    deps = [
        ":c_api",
        "//tensorflow/c:c_api",
        "//tensorflow/c:c_api_internal",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:framework_lite",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/common_runtime/eager:attr_builder",
        "//tensorflow/core/common_runtime/eager:context",
        "//tensorflow/core/common_runtime/eager:eager_executor",
        "//tensorflow/core/common_runtime/eager:eager_operation",
        "//tensorflow/core/common_runtime/eager:kernel_and_device",
        "//tensorflow/core/common_runtime/eager:tensor_handle",
        "//tensorflow/core/profiler/lib:profiler_session",
    ],
)

tf_cuda_library(
    name = "c_api_test_util",
    testonly = 1,
    srcs = ["c_api_test_util.cc"],
    hdrs = ["c_api_test_util.h"],
    visibility = [
        "//learning/brain:__subpackages__",
        "//tensorflow:__subpackages__",
    ],
    deps = [
        ":c_api",
        "//tensorflow/c:c_test_util",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
    ],
)

tf_cuda_cc_test(
    name = "c_api_test",
    size = "small",
    srcs = [
        "c_api_debug_test.cc",
        "c_api_test.cc",
    ],
    extra_copts = tfe_xla_copts(),
    tags = [
        "guitar",
        "multi_gpu",
    ],
    deps = [
        ":c_api",
        ":c_api_experimental",
        ":c_api_internal",
        ":c_api_test_util",
        "//tensorflow/c:c_test_util",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/strings",
    ],
)

tf_cuda_cc_test(
    name = "c_api_remote_test",
    size = "small",
    srcs = [
        "c_api_remote_test.cc",
    ],
    extra_copts = tfe_xla_copts(),
    tags = [
        "guitar",
        "multi_gpu",
        "no_oss",
    ],
    deps = [
        ":c_api",
        ":c_api_experimental",
        ":c_api_internal",
        ":c_api_test_util",
        "//tensorflow/c:c_test_util",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/distributed_runtime/rpc:grpc_server_lib",
        "@com_google_absl//absl/strings",
    ],
)

tf_cuda_library(
    name = "c_api_experimental",
    srcs = [
        "c_api_experimental.cc",
    ],
    hdrs = ["c_api_experimental.h"],
    copts = tf_copts() + tfe_xla_copts(),
    visibility = ["//visibility:public"],
    deps = select({
        "//tensorflow:android": [
            "//tensorflow/core:android_tensorflow_lib_lite",
        ],
        "//conditions:default": [
            ":c_api",
            ":c_api_internal",
            "//tensorflow/c:c_api",
            "//tensorflow/c:c_api_internal",
            "//tensorflow/core:core_cpu",
            "//tensorflow/core/common_runtime/eager:attr_builder",
            "//tensorflow/core/common_runtime/eager:context",
            "//tensorflow/core/common_runtime/eager:eager_executor",
            "//tensorflow/core/common_runtime/eager:execute",
            "//tensorflow/core/common_runtime/eager:kernel_and_device",
            "//tensorflow/core/common_runtime/eager:tensor_handle",
            "//tensorflow/core/common_runtime/eager:copy_to_device_node",
            "//tensorflow/core:core_cpu_internal",
            "//tensorflow/core:framework",
            "//tensorflow/core:framework_internal",
            "//tensorflow/core:lib",
            "//tensorflow/core:lib_internal",
            "//tensorflow/core:protos_all_cc",
        ],
    }) + select({
        "//tensorflow:with_xla_support": [
            "//tensorflow/compiler/tf2xla:xla_compiler",
            "//tensorflow/compiler/jit",
            "//tensorflow/compiler/jit:xla_device",
        ],
        "//conditions:default": [],
    }) + [
        "@com_google_absl//absl/memory",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/core/common_runtime/eager:eager_operation",
        "//tensorflow/core/distributed_runtime/eager:eager_client",
        "//tensorflow/core/distributed_runtime/rpc/eager:grpc_eager_client",
        "//tensorflow/core/distributed_runtime/rpc:grpc_channel",
        "//tensorflow/core/distributed_runtime/rpc:grpc_server_lib",
        "//tensorflow/core/distributed_runtime/rpc:grpc_worker_cache",
        "//tensorflow/core/distributed_runtime/rpc:grpc_worker_service",
        "//tensorflow/core/distributed_runtime/rpc:rpc_rendezvous_mgr",
        "//tensorflow/core/distributed_runtime:remote_device",
        "//tensorflow/core/distributed_runtime:server_lib",
        "//tensorflow/core/distributed_runtime:worker_env",
        "//tensorflow/core/profiler/rpc:profiler_server",
        "//tensorflow/core/profiler/rpc/client:capture_profile",
        "//tensorflow/core:gpu_runtime",
    ],
    alwayslink = 1,
)

tf_cuda_cc_test(
    name = "c_api_experimental_test",
    size = "small",
    srcs = [
        "c_api_experimental_test.cc",
    ],
    args = ["--heap_check=local"],
    extra_copts = tfe_xla_copts(),
    linkstatic = tf_kernel_tests_linkstatic(),
    tags = tf_cuda_tests_tags() + ["nomac"],
    deps = [
        ":c_api",
        ":c_api_experimental",
        ":c_api_test_util",
        "//tensorflow/c:c_test_util",
        "//tensorflow/cc/profiler",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "tape",
    hdrs = ["tape.h"],
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
)

filegroup(
    name = "headers",
    srcs = ["c_api.h"],
    visibility = ["//tensorflow:__subpackages__"],
)

# TODO(karllessard): only used by //tensorflow/core:mobile_srcs_only_runtime
# right now, remove this public rule when no longer needed (it should be
# replaced by TF Lite)
filegroup(
    name = "srcs",
    srcs = glob(
        [
            "*.cc",
            "*.h",
        ],
        exclude = [
            "c_api_experimental.cc",
            "*test*",
        ],
    ),
    visibility = ["//visibility:public"],
)

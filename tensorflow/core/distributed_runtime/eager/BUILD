load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//tensorflow:tensorflow.default.bzl", "filegroup", "tf_grpc_cc_dependencies")
load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow:internal",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "remote_tensor_handle",
    hdrs = ["remote_tensor_handle.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "cluster_function_library_runtime",
    srcs = [
        "cluster_function_library_runtime.cc",
    ],
    hdrs = [
        "cluster_function_library_runtime.h",
    ],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":eager_client",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/common_runtime/eager:context",
        "//tensorflow/core/common_runtime/eager:eager_operation",
        "//tensorflow/core/distributed_runtime:call_options",
        "//tensorflow/core/distributed_runtime:worker_session",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "destroy_tensor_handle_node",
    hdrs = ["destroy_tensor_handle_node.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":eager_client",
        "//tensorflow/core/common_runtime/eager:eager_executor",
        "//tensorflow/core/protobuf:eager_service_proto_cc",
    ],
)

cc_library(
    name = "eager_client",
    hdrs = ["eager_client.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/distributed_runtime:call_options",
        "//tensorflow/core/protobuf:eager_service_proto_cc",
    ],
)

cc_library(
    name = "remote_execute_node",
    srcs = ["remote_execute_node.cc"],
    hdrs = ["remote_execute_node.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":eager_client",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/common_runtime/eager:eager_executor",
        "//tensorflow/core/common_runtime/eager:shape_inference",
        "//tensorflow/core/common_runtime/eager:tensor_handle",
        "//tensorflow/core/protobuf:eager_service_proto_cc",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "eager_service_impl",
    srcs = ["eager_service_impl.cc"],
    hdrs = [
        "eager_service_impl.h",
    ],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":cluster_function_library_runtime",
        ":remote_mgr",
        ":remote_tensor_handle",
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "//tensorflow/c/eager:immediate_execution_distributed_manager",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/common_runtime/eager:context",
        "//tensorflow/core/common_runtime/eager:context_distributed_manager",
        "//tensorflow/core/common_runtime/eager:core",
        "//tensorflow/core/common_runtime/eager:eager_operation",
        "//tensorflow/core/common_runtime/eager:execute",
        "//tensorflow/core/distributed_runtime:message_wrappers",
        "//tensorflow/core/distributed_runtime:session_mgr",
        "//tensorflow/core/distributed_runtime:worker_cache",
        "//tensorflow/core/distributed_runtime:worker_env",
        "//tensorflow/core/profiler/lib:traceme",
        "//tensorflow/tsl/distributed_runtime/preemption:preemption_notifier",
    ] + tf_grpc_cc_dependencies(),
)

tf_cc_test(
    name = "eager_service_impl_test",
    srcs = ["eager_service_impl_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":cluster_function_library_runtime",
        ":eager_service_impl",
        ":remote_mgr",
        "//tensorflow/c:c_api_internal",
        "//tensorflow/c:tf_tensor_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensorflow",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/common_runtime/eager:kernel_and_device",
        "//tensorflow/core/common_runtime/eager:tensor_handle",
        "//tensorflow/core/distributed_runtime:session_mgr",
        "//tensorflow/core/distributed_runtime:test_utils",
        "//tensorflow/core/distributed_runtime:worker_env",
        "//tensorflow/core/distributed_runtime/rpc:rpc_rendezvous_mgr",
        "//tensorflow/core/protobuf:eager_service_proto_cc",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:variant",
    ],
)

cc_library(
    name = "remote_mgr",
    srcs = [
        "remote_mgr.cc",
    ],
    hdrs = [
        "remote_mgr.h",
    ],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    visibility = ["//tensorflow:internal"],
    deps = [
        ":remote_tensor_handle",
        "//tensorflow/core:lib",
        "//tensorflow/core/common_runtime/eager:eager_executor",
        "//tensorflow/core/common_runtime/eager:tensor_handle",
        "//tensorflow/core/platform:error_payloads",
    ],
)

tf_cc_test(
    name = "remote_mgr_test",
    size = "small",
    srcs = ["remote_mgr_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":remote_mgr",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/common_runtime/eager:core",
        "//tensorflow/core/common_runtime/eager:tensor_handle",
        "//tensorflow/core/platform:error_payloads",
    ],
)

cc_library(
    name = "remote_tensor_handle_data",
    srcs = ["remote_tensor_handle_data.cc"],
    hdrs = ["remote_tensor_handle_data.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":destroy_tensor_handle_node",
        ":eager_client",
        "//tensorflow/core:lib",
        "//tensorflow/core/common_runtime/eager:context",
        "//tensorflow/core/profiler/lib:traceme",
    ],
)

cc_library(
    name = "remote_copy_node",
    srcs = [
        "remote_copy_node.cc",
    ],
    hdrs = [
        "remote_copy_node.h",
    ],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    visibility = ["//tensorflow:internal"],
    deps = [
        ":remote_mgr",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/common_runtime/eager:attr_builder",
        "//tensorflow/core/common_runtime/eager:eager_executor",
        "//tensorflow/core/common_runtime/eager:eager_operation",
        "//tensorflow/core/common_runtime/eager:tensor_handle",
        "@com_google_absl//absl/types:optional",
    ],
)

filegroup(
    name = "pywrap_required_hdrs",
    srcs = [
        "eager_client.h",
        "remote_tensor_handle.h",
        "remote_tensor_handle_data.h",
    ],
    visibility = [
        "//tensorflow/core/function/runtime_client:__pkg__",
        "//tensorflow/python:__subpackages__",
    ],
)

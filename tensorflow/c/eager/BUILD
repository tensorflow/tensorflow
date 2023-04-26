# Experimental extensions to the C API for eager execution of kernels.

load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
    "tf_copts",
    "tf_cuda_cc_test",
    "tf_cuda_library",
)
load("//tensorflow:tensorflow.default.bzl", "cc_header_only_library", "filegroup", "internal_tfrt_deps")
load(
    "//tensorflow/core/platform:build_config_root.bzl",
    "tf_cuda_tests_tags",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

tf_cuda_library(
    name = "c_api",
    srcs = [
        "c_api.cc",
        "c_api_debug.cc",
        "c_api_experimental.h",
        "c_api_internal.h",
        "c_api_unified_experimental.h",
    ],
    hdrs = ["c_api.h"],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = select({
        "//tensorflow:android": [
            "//tensorflow/core:portable_tensorflow_lib_lite",
        ],
        "//conditions:default": [
            ":immediate_execution_context",
            ":immediate_execution_distributed_manager",
            ":immediate_execution_operation",
            ":immediate_execution_tensor_handle",
            ":tfe_cancellation_manager_internal",
            ":tfe_context_internal",
            ":tfe_executor_internal",
            ":tfe_monitoring_internal",
            ":tfe_op_attrs_internal",
            ":tfe_op_internal",
            ":tfe_tensor_debug_info_internal",
            ":tfe_tensorhandle_internal",
            "//tensorflow/c:c_api",
            "//tensorflow/c:c_api_internal",
            "//tensorflow/c:tf_buffer",
            "//tensorflow/c:tf_buffer_internal",
            "//tensorflow/c:tf_status_internal",
            "//tensorflow/c:tf_tensor_internal",
            "//tensorflow/core:core_cpu",
            "//tensorflow/core:core_cpu_internal",
            "//tensorflow/core:framework",
            "//tensorflow/core:framework_internal",
            "//tensorflow/core:lib",
            "//tensorflow/core:lib_internal",
            "//tensorflow/core:protos_all_cc",
            "//tensorflow/core/common_runtime/eager:attr_builder",
            "//tensorflow/core/common_runtime/eager:context",
            "//tensorflow/core/common_runtime/eager:context_distributed_manager",
            "//tensorflow/core/common_runtime/eager:core",
            "//tensorflow/core/common_runtime/eager:custom_device",
            "//tensorflow/core/common_runtime/eager:eager_executor",
            "//tensorflow/core/common_runtime/eager:execute",
            "//tensorflow/core/common_runtime/eager:placement_utils",
            "//tensorflow/core/common_runtime/eager:tensor_handle",
            "//tensorflow/core/profiler/lib:traceme",
            "@com_google_absl//absl/algorithm:container",
            "@com_google_absl//absl/types:span",
            "@com_google_absl//absl/types:variant",
        ],
    }) + [
        ":abstract_tensor_handle",
        "//tensorflow/core:gpu_runtime",
        "//tensorflow/core/common_runtime/eager:eager_operation",
        "//tensorflow/core/distributed_runtime:remote_device",
        "//tensorflow/core/distributed_runtime:server_lib",
        "//tensorflow/core/distributed_runtime:worker_env",
        "//tensorflow/core/distributed_runtime:worker_interface",
        "//tensorflow/core/distributed_runtime/eager:cluster_function_library_runtime",
        "//tensorflow/core/distributed_runtime/eager:eager_client",
        "//tensorflow/core/distributed_runtime/eager:remote_mgr",
        "//tensorflow/core/distributed_runtime/rpc:grpc_channel",
        "//tensorflow/core/distributed_runtime/rpc:grpc_server_lib",
        "//tensorflow/core/distributed_runtime/rpc:grpc_worker_cache",
        "//tensorflow/core/distributed_runtime/rpc:grpc_worker_service",
        "//tensorflow/core/distributed_runtime/rpc:rpc_rendezvous_mgr",
        "//tensorflow/core/distributed_runtime/rpc/eager:grpc_eager_client",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings:str_format",
    ] + internal_tfrt_deps(),
    alwayslink = 1,
)

filegroup(
    name = "pywrap_required_hdrs",
    srcs = [
        "abstract_context.h",
        "abstract_function.h",
        "abstract_op_attrs.h",
        "abstract_operation.h",
        "abstract_tensor_handle.h",
        "c_api.h",
        "c_api_experimental.h",
        "c_api_internal.h",
        "c_api_unified_experimental.h",
        "c_api_unified_experimental_internal.h",
        "dlpack.h",
        "gradients.h",
        "gradients_internal.h",
        "immediate_execution_context.h",
        "immediate_execution_distributed_manager.h",
        "immediate_execution_operation.h",
        "immediate_execution_tensor_handle.h",
        "tape.h",
        "tfe_cancellation_manager_internal.h",
        "tfe_context_internal.h",
        "tfe_executor_internal.h",
        "tfe_monitoring_internal.h",
        "tfe_op_attrs_internal.h",
        "tfe_tensor_debug_info_internal.h",
        "tfe_tensorhandle_internal.h",
    ],
    visibility = [
        "//tensorflow/core/function/runtime_client:__pkg__",
        "//tensorflow/python:__subpackages__",
    ],
)

cc_library(
    name = "c_api_internal",
    hdrs = [
        "c_api_experimental.h",
        "c_api_internal.h",
    ],
    visibility = ["//tensorflow:internal"],
    deps = [
        ":c_api",
        ":tfe_cancellation_manager_internal",
        ":tfe_executor_internal",
        ":tfe_monitoring_internal",
        ":tfe_op_attrs_internal",
        ":tfe_tensor_debug_info_internal",
        "//tensorflow/c:c_api_internal",
    ],
)

cc_library(
    name = "c_api_unified_internal",
    hdrs = [
        "c_api_unified_experimental_internal.h",
    ],
    visibility = [
        "//tensorflow:internal",
    ],
    deps = [
        ":abstract_context",
        ":abstract_operation",
        ":abstract_tensor_handle",
        ":c_api",
        ":c_api_experimental",
        "//tensorflow/c:c_api_internal",
        "//tensorflow/c:conversion_macros",
        "//tensorflow/c:tf_status",
        "//tensorflow/core:framework",
        "//tensorflow/core/platform:casts",
        "//tensorflow/core/platform:types",
    ],
)

cc_library(
    name = "tracing_utils",
    srcs = ["tracing_utils.cc"],
    hdrs = [
        "tracing_utils.h",
    ],
    visibility = [
        "//tensorflow:internal",
    ],
    deps = [
        ":abstract_operation",
        ":c_api_unified_internal",
        "//tensorflow/c/experimental/gradients/tape:tape_operation",
        "//tensorflow/core/lib/llvm_rtti",
        "//tensorflow/core/platform:errors",
    ],
)

cc_library(
    name = "gradients_internal",
    srcs = [
        "gradients.cc",
    ],
    hdrs = [
        "gradients.h",
        "gradients_internal.h",
    ],
    visibility = [
        "//tensorflow:internal",
    ],
    deps = [
        ":abstract_context",
        ":abstract_tensor_handle",
        ":c_api_unified_internal",
        ":tape",
        "//tensorflow/core/common_runtime/eager:attr_builder",
        "//tensorflow/core/lib/llvm_rtti",
        "//tensorflow/core/platform:errors",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "unified_api_testutil",
    testonly = 1,
    srcs = [
        "unified_api_testutil.cc",
    ],
    hdrs = [
        "unified_api_testutil.h",
    ],
    visibility = [
        "//tensorflow:internal",
    ],
    deps = [
        ":abstract_context",
        ":abstract_tensor_handle",
        ":c_api_experimental",
        ":c_api_test_util",
        ":c_api_unified_internal",
        "//tensorflow/c:tf_status",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/c:tf_tensor",
        "//tensorflow/core:framework",
        "//tensorflow/core/lib/llvm_rtti",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

tf_cuda_cc_test(
    name = "gradients_test",
    size = "small",
    srcs = [
        "gradients_test.cc",
    ],
    args = ["--heap_check="],
    tags = tf_cuda_tests_tags() + ["nomac"],
    deps = [
        ":abstract_context",
        ":abstract_tensor_handle",
        ":c_api_experimental",
        ":c_api_test_util",
        ":c_api_unified_internal",
        ":gradients_internal",
        ":unified_api_testutil",
        "//tensorflow/c:c_api",
        "//tensorflow/c:c_test_util",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/c/experimental/gradients:array_grad",
        "//tensorflow/c/experimental/gradients:math_grad",
        "//tensorflow/c/experimental/gradients:not_differentiable",
        "//tensorflow/c/experimental/gradients/tape:tape_context",
        "//tensorflow/c/experimental/ops",
        "//tensorflow/compiler/mlir/tensorflow/c:mlir_c_api_registration",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/lib/llvm_rtti",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

tf_cuda_cc_test(
    name = "unified_api_test",
    size = "small",
    srcs = [
        "unified_api_test.cc",
    ],
    args = ["--heap_check="],
    tags = tf_cuda_tests_tags() + ["no_cuda_asan"],  # b/173654156
    deps = [
        ":c_api_experimental",
        ":c_api_unified_internal",
        ":unified_api_testutil",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/compiler/mlir/tensorflow/c:mlir_c_api_registration",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/lib/llvm_rtti",
        "//tensorflow/core/platform:errors",
    ],
)

cc_library(
    name = "gradient_checker",
    testonly = 1,
    srcs = [
        "gradient_checker.cc",
    ],
    hdrs = [
        "gradient_checker.h",
    ],
    visibility = [
        "//tensorflow:internal",
    ],
    deps = [
        ":abstract_tensor_handle",
        ":unified_api_testutil",
        "//tensorflow/c:tf_tensor_internal",
        "//tensorflow/c/experimental/ops:math_ops",
        "@com_google_absl//absl/types:span",
    ],
)

tf_cuda_cc_test(
    name = "gradient_checker_test",
    size = "small",
    srcs = [
        "gradient_checker_test.cc",
    ],
    args = ["--heap_check="],
    tags = tf_cuda_tests_tags() + [
        "no_cuda_asan",  # b/175330074
    ],
    deps = [
        ":abstract_tensor_handle",
        ":c_api_experimental",
        ":gradient_checker",
        ":unified_api_testutil",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/c:tf_tensor_internal",
        "//tensorflow/c/experimental/ops",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform:tensor_float_32_utils",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "abstract_tensor_handle",
    srcs = ["abstract_tensor_handle.cc"],
    hdrs = ["abstract_tensor_handle.h"],
    visibility = [
        "//tensorflow:internal",
    ],
    deps = select({
        "//tensorflow:android": [
            "//tensorflow/core:portable_tensorflow_lib_lite",
        ],
        "//conditions:default": [
            "//tensorflow/core:framework",
            "//tensorflow/core:protos_all_cc",
            "//tensorflow/core/platform:refcount",
            "//tensorflow/core/platform:status",
        ],
    }),
)

cc_library(
    name = "immediate_execution_tensor_handle",
    srcs = ["immediate_execution_tensor_handle.cc"],
    hdrs = ["immediate_execution_tensor_handle.h"],
    visibility = [
        "//tensorflow:internal",
    ],
    deps = [
        ":abstract_tensor_handle",
        "//tensorflow/c:tensor_interface",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "abstract_op_attrs",
    hdrs = ["abstract_op_attrs.h"],
    visibility = [
        "//tensorflow:internal",
    ],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/container:inlined_vector",
    ],
)

cc_library(
    name = "abstract_operation",
    hdrs = ["abstract_operation.h"],
    visibility = [
        "//tensorflow:internal",
    ],
    deps = [
        ":abstract_tensor_handle",
        "//tensorflow/c:tensor_interface",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "immediate_execution_operation",
    hdrs = ["immediate_execution_operation.h"],
    visibility = [
        "//tensorflow:internal",
    ],
    deps = [
        ":abstract_operation",
        ":abstract_tensor_handle",
        ":immediate_execution_tensor_handle",
        "//tensorflow/c:tensor_interface",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/util:managed_stack_trace",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "abstract_context",
    hdrs = ["abstract_context.h"],
    visibility = [
        "//tensorflow:internal",
    ],
    deps = [
        ":abstract_function",
        ":abstract_operation",
    ],
)

cc_library(
    name = "abstract_function",
    hdrs = ["abstract_function.h"],
    visibility = [
        "//tensorflow:internal",
    ],
    deps = [
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:intrusive_ptr",
        "//tensorflow/core/platform:refcount",
        "//tensorflow/core/platform:status",
    ],
)

cc_library(
    name = "immediate_execution_distributed_manager",
    hdrs = ["immediate_execution_distributed_manager.h"],
    visibility = [
        "//tensorflow:internal",
    ],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "immediate_execution_context",
    hdrs = ["immediate_execution_context.h"],
    visibility = [
        "//tensorflow:internal",
    ],
    deps = [
        ":abstract_context",
        ":immediate_execution_distributed_manager",
        ":immediate_execution_operation",
        ":immediate_execution_tensor_handle",
        "//tensorflow/c:tensor_interface",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "tfe_context_internal",
    hdrs = ["tfe_context_internal.h"],
    visibility = [
        "//tensorflow:internal",
    ],
    deps = [
        ":immediate_execution_context",
        "//tensorflow/c:conversion_macros",
    ],
)

cc_library(
    name = "tfe_cancellation_manager_internal",
    hdrs = ["tfe_cancellation_manager_internal.h"],
    visibility = [
        "//tensorflow:internal",
    ],
    deps = [
        "//tensorflow/c:conversion_macros",
        "//tensorflow/core:framework",
    ],
)

cc_library(
    name = "tfe_executor_internal",
    hdrs = ["tfe_executor_internal.h"],
    visibility = ["//visibility:private"],
    deps = [
        "//tensorflow/core/common_runtime/eager:eager_executor",
    ],
)

cc_library(
    name = "tfe_monitoring_internal",
    hdrs = ["tfe_monitoring_internal.h"],
    visibility = ["//visibility:private"],
    deps = [
        "//tensorflow/core:lib",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "tfe_op_attrs_internal",
    hdrs = ["tfe_op_attrs_internal.h"],
    visibility = [
        "//tensorflow:internal",
    ],
    deps = [
        ":abstract_op_attrs",
        "//tensorflow/c:conversion_macros",
        "//tensorflow/c:tf_status",
        "//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "tfe_op_internal",
    hdrs = ["tfe_op_internal.h"],
    visibility = [
        "//tensorflow:internal",
    ],
    deps = [
        ":immediate_execution_operation",
        "//tensorflow/c:conversion_macros",
    ],
)

cc_library(
    name = "tfe_tensor_debug_info_internal",
    hdrs = ["tfe_tensor_debug_info_internal.h"],
    visibility = ["//visibility:private"],
    deps = [
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "tfe_tensorhandle_internal",
    hdrs = ["tfe_tensorhandle_internal.h"],
    visibility = [
        "//tensorflow:internal",
    ],
    deps = [
        ":immediate_execution_tensor_handle",
        "//tensorflow/c:conversion_macros",
    ],
)

cc_header_only_library(
    name = "tfe_tensorhandle_internal_hdrs_only",
    extra_deps = [
        "@com_google_absl//absl/strings",
    ],
    visibility = [
        "//tensorflow:internal",
    ],
    deps = [
        ":tfe_tensorhandle_internal",
    ],
)

cc_header_only_library(
    name = "tfe_cancellationmanager_internal_hdrs_only",
    extra_deps = [
        "@com_google_absl//absl/strings",
    ],
    visibility = [
        "//tensorflow:internal",
    ],
    deps = [
        ":tfe_cancellation_manager_internal",
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
        ":c_api_experimental",
        "//tensorflow/c:c_test_util",
        "//tensorflow/c:tf_datatype",
        "//tensorflow/c:tf_tensor",
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
    tags = [
        "no_cuda_asan",  # TODO(b/181771536)
        "guitar",
        # "multi_gpu",  b/180748118
    ],
    deps = [
        ":c_api",
        ":c_api_experimental",
        ":c_api_internal",
        ":c_api_test_util",
        ":tfe_op_internal",
        ":tfe_tensorhandle_internal",
        "@com_google_absl//absl/strings",
        "//tensorflow/c:c_test_util",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/common_runtime/eager:eager_operation",
        "//tensorflow/core/common_runtime/eager:tensor_handle",
        "//tensorflow/core/distributed_runtime/rpc:grpc_server_lib",
        # copybara:uncomment_begin
        # "//tensorflow/core/tfrt/eager:c_api_tfrt",
        # "@tf_runtime//backends/cpu:tf_ops_alwayslink",
        # copybara:uncomment_end
    ],
)

tf_cuda_library(
    name = "c_api_remote_test_util",
    testonly = 1,
    srcs = ["c_api_remote_test_util.cc"],
    hdrs = ["c_api_remote_test_util.h"],
    visibility = ["//visibility:private"],
    deps = [
        ":c_api",
        ":c_api_internal",
        ":c_api_test_util",
        ":tfe_tensorhandle_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core/common_runtime/eager:tensor_handle",
        "//tensorflow/core/distributed_runtime/rpc:grpc_server_lib",
        "@com_google_absl//absl/strings",
    ],
)

tf_cuda_cc_test(
    name = "c_api_remote_test",
    size = "small",
    srcs = [
        "c_api_remote_test.cc",
    ],
    # TODO(b/136478427): Figure out how to correctly shut the server down
    args = ["--heap_check="],
    tags = [
        "no_windows",
    ],
    deps = [
        ":c_api",
        ":c_api_experimental",
        ":c_api_internal",
        ":c_api_remote_test_util",
        ":c_api_test_util",
        ":tfe_tensorhandle_internal",
        "//tensorflow/c:c_test_util",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/common_runtime:function_optimization_registry",
        "//tensorflow/core/common_runtime/eager:eager_operation",
        "//tensorflow/core/distributed_runtime/rpc:grpc_server_lib",
        "@com_google_absl//absl/strings",
    ],
)

tf_cuda_cc_test(
    name = "c_api_remote_function_test",
    size = "small",
    srcs = [
        "c_api_remote_function_test.cc",
    ],
    # TODO(b/136478427): Figure out how to correctly shut the server down
    args = ["--heap_check="],
    tags = [
        "no_windows",
    ],
    deps = [
        ":c_api_remote_test_util",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cuda_cc_test(
    name = "c_api_distributed_test",
    size = "small",
    srcs = [
        "c_api_distributed_test.cc",
    ],
    # TODO(b/136478427): Figure out how to correctly shut the server down
    args = ["--heap_check="],
    tags = [
        "no_oss",  # TODO(b/200848572)
        "no_windows",
    ],
    deps = [
        ":c_api",
        ":c_api_experimental",
        ":c_api_internal",
        ":c_api_test_util",
        ":tfe_tensorhandle_internal",
        "//tensorflow/c:c_test_util",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/common_runtime:function_optimization_registry",
        "//tensorflow/core/common_runtime:optimization_registry",
        "//tensorflow/core/common_runtime/eager:eager_operation",
        "//tensorflow/core/distributed_runtime/rpc:grpc_server_lib",
        "@com_google_absl//absl/strings",
    ],
)

tf_cuda_cc_test(
    name = "c_api_cluster_test",
    size = "small",
    srcs = [
        "c_api_cluster_test.cc",
    ],
    # TODO(b/136478427): Figure out how to correctly shut the server down
    args = ["--heap_check="],
    tags = [
        "no_windows",
    ],
    deps = [
        ":c_api",
        ":c_api_experimental",
        ":c_api_internal",
        ":c_api_test_util",
        ":tfe_tensorhandle_internal",
        "//tensorflow/c:c_test_util",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/common_runtime/eager:eager_operation",
        "//tensorflow/core/distributed_runtime/rpc:grpc_server_lib",
        "//tensorflow/core/platform:env",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "graph_function",
    srcs = ["graph_function.cc"],
    hdrs = [
        "graph_function.h",
    ],
    visibility = [
        "//tensorflow:internal",
    ],
    deps = [
        ":abstract_function",
        "//tensorflow/core/lib/llvm_rtti",
        "//tensorflow/core/platform:errors",
    ],
)

tf_cuda_library(
    name = "c_api_experimental",
    srcs = [
        "c_api_experimental.cc",
        "c_api_unified_experimental.cc",
        "c_api_unified_experimental_eager.cc",
        "c_api_unified_experimental_graph.cc",
        "c_api_unified_experimental_internal.h",
    ],
    hdrs = [
        "c_api_experimental.h",
        "c_api_unified_experimental.h",
    ],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = select({
        "//tensorflow:android": [
            "//tensorflow/core:portable_tensorflow_lib_lite",
        ],
        "//conditions:default": [
            ":abstract_context",
            ":abstract_operation",
            ":abstract_tensor_handle",
            ":c_api",
            ":c_api_internal",
            ":graph_function",
            ":immediate_execution_context",
            ":immediate_execution_tensor_handle",
            ":tfe_context_internal",
            ":tfe_op_internal",
            ":tfe_tensorhandle_internal",
            "//tensorflow/c:c_api",
            "//tensorflow/c:c_api_internal",
            "//tensorflow/c:conversion_macros",
            "//tensorflow/core:core_cpu",
            "//tensorflow/core:core_cpu_internal",
            "//tensorflow/core:framework",
            "//tensorflow/core:framework_internal",
            "//tensorflow/core:lib",
            "//tensorflow/core:lib_internal",
            "//tensorflow/core:protos_all_cc",
            "//tensorflow/core/common_runtime/eager:attr_builder",
            "//tensorflow/core/common_runtime/eager:context",
            "//tensorflow/core/common_runtime/eager:copy_to_device_node",
            "//tensorflow/core/common_runtime/eager:eager_executor",
            "//tensorflow/core/common_runtime/eager:eager_operation",
            "//tensorflow/core/common_runtime/eager:execute",
            "//tensorflow/core/common_runtime/eager:kernel_and_device",
            "//tensorflow/core/common_runtime/eager:tensor_handle",
            "//tensorflow/core/lib/llvm_rtti",
            "@com_google_absl//absl/types:variant",
        ],
    }) + select({
        "//tensorflow:with_xla_support": [
            "//tensorflow/compiler/jit",
            "//tensorflow/compiler/jit:xla_device",
            "//tensorflow/compiler/tf2xla:xla_compiler",
        ],
        "//conditions:default": [],
    }) + [
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/core:gpu_runtime",
        "//tensorflow/core/distributed_runtime:remote_device",
        "//tensorflow/core/distributed_runtime:server_lib",
        "//tensorflow/core/distributed_runtime:worker_env",
        "//tensorflow/core/distributed_runtime/coordination:coordination_service_error_util",
        "//tensorflow/core/distributed_runtime/eager:eager_client",
        "//tensorflow/core/distributed_runtime/rpc:grpc_channel",
        "//tensorflow/core/distributed_runtime/rpc:grpc_server_lib",
        "//tensorflow/core/distributed_runtime/rpc:grpc_worker_cache",
        "//tensorflow/core/distributed_runtime/rpc:grpc_worker_service",
        "//tensorflow/core/distributed_runtime/rpc:rpc_rendezvous_mgr",
        "//tensorflow/core/distributed_runtime/rpc/eager:grpc_eager_client",
        "//tensorflow/tsl/distributed_runtime/coordination:coordination_service_agent",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
    alwayslink = 1,
)

cc_library(
    name = "c_api_experimental_reader",
    testonly = True,
    srcs = [
        "c_api_experimental_reader.cc",
    ],
    hdrs = [
        "c_api_experimental_reader.h",
        "tfe_monitoring_reader_internal.h",
    ],
    visibility = ["//tensorflow:internal"],
    deps = [
        ":c_api",
        "//tensorflow/c:c_api",
        "//tensorflow/core/lib/monitoring:cell_reader",
        "@com_google_absl//absl/memory",
    ],
    alwayslink = 1,
)

tf_cuda_cc_test(
    name = "c_api_experimental_test",
    size = "small",
    srcs = [
        "c_api_experimental_test.cc",
    ],
    args = ["--heap_check="],
    tags = tf_cuda_tests_tags() + ["nomac"],
    deps = [
        ":c_api",
        ":c_api_experimental",
        ":c_api_test_util",
        "//tensorflow/c:c_test_util",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform:status",
        "@com_google_absl//absl/strings",
    ],
)

tf_cuda_cc_test(
    name = "c_api_experimental_reader_test",
    size = "small",
    srcs = [
        "c_api_experimental_reader_test.cc",
    ],
    args = ["--heap_check="],
    tags = tf_cuda_tests_tags() + ["nomac"],
    deps = [
        ":c_api",
        ":c_api_experimental",
        ":c_api_experimental_reader",
        ":c_api_test_util",
        "//tensorflow/c:c_test_util",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform:status",
        "@com_google_absl//absl/strings",
    ],
)

tf_cuda_cc_test(
    name = "c_api_unified_experimental_test",
    size = "small",
    srcs = [
        "c_api_unified_experimental_internal.h",
        "c_api_unified_experimental_test.cc",
    ],
    args = ["--heap_check="],
    tags = tf_cuda_tests_tags() + ["nomac"],
    deps = [
        ":abstract_context",
        ":abstract_operation",
        ":abstract_tensor_handle",
        ":c_api",
        ":c_api_experimental",
        ":c_api_test_util",
        "//tensorflow/c:c_api",
        "//tensorflow/c:c_test_util",
        "//tensorflow/c:conversion_macros",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/compiler/mlir/tensorflow/c:mlir_c_api_registration",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "custom_device_testutil",
    testonly = True,
    srcs = ["custom_device_testutil.cc"],
    hdrs = ["custom_device_testutil.h"],
    visibility = ["//tensorflow:internal"],
    deps = [
        ":c_api",
        ":c_api_experimental",
        ":c_api_test_util",
        "//tensorflow/c:c_api",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
    ],
)

tf_cc_test(
    name = "custom_device_test",
    size = "small",
    srcs = [
        "custom_device_test.cc",
    ],
    deps = [
        ":c_api",
        ":c_api_experimental",
        ":c_api_test_util",
        ":custom_device_testutil",
        "//tensorflow/c:c_api",
        "//tensorflow/c:c_test_util",
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
        "//tensorflow/core/config:flag_defs",
        "//tensorflow/core/config:flags",
    ],
)

filegroup(
    name = "headers",
    srcs = [
        "c_api.h",
        "c_api_experimental.h",
        "dlpack.h",
    ],
    visibility = ["//tensorflow:__subpackages__"],
)

filegroup(
    name = "pywrap_headers_monitoring_reader",
    srcs = [
        "c_api_experimental_reader.h",
        "tfe_monitoring_reader_internal.h",
    ],
    visibility = ["//tensorflow:__subpackages__"],
)

filegroup(
    name = "headers_monitoring_reader",
    srcs = [
        "c_api_experimental_reader.h",
    ],
    visibility = ["//tensorflow:__subpackages__"],
)

cc_library(
    name = "dlpack",
    srcs = ["dlpack.cc"],
    hdrs = ["dlpack.h"],
    copts = tf_copts() + [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        ":c_api",
        ":c_api_experimental",
        ":tfe_tensorhandle_internal",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/c:tf_status_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core/common_runtime/eager:tensor_handle",
        "@dlpack",
    ],
    alwayslink = 1,
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
            "c_api_experimental_reader.cc",
            "c_api_experimental_reader.h",
            "tfe_monitoring_reader_internal.h",
            "c_api_unified_experimental.cc",
            "c_api_unified_experimental_eager.cc",
            "c_api_unified_experimental_graph.cc",
            "c_api_unified_experimental_internal.h",
            "graph_function.h",
            "graph_function.cc",
            "gradient_checker.cc",
            "gradient_checker.h",
            "gradients.cc",  # Uses RTTI.
            "tracing_utils.h",
            "tracing_utils.cc",
            "*test*",
            "*dlpack*",
        ],
    ),
    visibility = ["//visibility:public"],
)

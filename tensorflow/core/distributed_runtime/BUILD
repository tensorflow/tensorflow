# Description:
# A distributed runtime for TensorFlow, which allows graph execution
# to be distributed and performed in parallel across multiple
# processes.

load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load(
    "//tensorflow:tensorflow.bzl",
    "if_google",
    "tf_cc_test",
    "tf_copts",
    "tf_cuda_library",
)
load(
    "//tensorflow:tensorflow.default.bzl",
    "filegroup",
    "tf_cuda_cc_test",
    "tf_grpc_cc_dependencies",
)
load(
    "//tensorflow/core/platform:build_config_root.bzl",
    "tf_cuda_tests_tags",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = if_google(
        ["//tensorflow:internal"],
        ["//visibility:public"],
    ),
    licenses = ["notice"],
)

exports_files(
    ["server_lib.h"],
    visibility = ["//tensorflow:internal"],
)

filegroup(
    name = "c_srcs",
    data = glob([
        "**/*.cc",
        "**/*.h",
    ]),
)

cc_library(
    name = "partial_run_mgr",
    srcs = ["partial_run_mgr.cc"],
    hdrs = ["partial_run_mgr.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":worker_interface",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "partial_run_mgr_test",
    size = "small",
    srcs = ["partial_run_mgr_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":partial_run_mgr",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "message_wrappers",
    srcs = ["message_wrappers.cc"],
    hdrs = ["message_wrappers.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/protobuf:master_proto_cc",
        "//tensorflow/core/protobuf:worker_proto_cc",
        "@com_google_absl//absl/status",
    ],
)

tf_cc_test(
    name = "message_wrappers_test",
    size = "small",
    srcs = ["message_wrappers_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":message_wrappers",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/framework:tensor_testutil",
    ],
)

cc_library(
    name = "worker_session",
    srcs = ["worker_session.cc"],
    hdrs = ["worker_session.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":graph_mgr",
        ":worker_cache",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

cc_library(
    name = "cluster_function_library_runtime",
    srcs = ["cluster_function_library_runtime.cc"],
    hdrs = ["cluster_function_library_runtime.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":worker_cache",
        ":worker_interface",
        ":worker_session",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/protobuf:worker_proto_cc",
        "@com_google_absl//absl/types:optional",
    ],
)

tf_cc_test(
    name = "cluster_function_library_runtime_test",
    srcs = ["cluster_function_library_runtime_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    tags = [
        "no_windows",
    ],
    deps = [
        ":cluster_function_library_runtime",
        ":worker_session",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/distributed_runtime/rpc:grpc_channel",
        "//tensorflow/core/distributed_runtime/rpc:grpc_testlib",
        "//tensorflow/core/distributed_runtime/rpc:grpc_worker_cache",
        "//tensorflow/core/kernels:cast_op",
        "//tensorflow/core/kernels:cwise_op",
    ],
)

cc_library(
    name = "session_mgr",
    srcs = ["session_mgr.cc"],
    hdrs = ["session_mgr.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":cluster_function_library_runtime",
        ":error_payloads",
        ":graph_mgr",
        ":remote_device",
        ":worker_cache_wrapper",
        ":worker_session",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/activity_watcher",
        "//tensorflow/core/protobuf:worker_proto_cc",
        "//tensorflow/tsl/distributed_runtime/coordination:coordination_service",
        "//tensorflow/tsl/distributed_runtime/coordination:coordination_service_agent",
        "//tensorflow/tsl/distributed_runtime/coordination:coordination_service_rpc_handler",
        "//tensorflow/tsl/protobuf:coordination_service_proto_cc",
    ],
)

tf_cc_test(
    name = "session_mgr_test",
    size = "small",
    srcs = ["session_mgr_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":error_payloads",
        ":session_mgr",
        ":worker_env",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/distributed_runtime/rpc:rpc_rendezvous_mgr",
    ],
)

cc_library(
    name = "worker_env",
    hdrs = ["worker_env.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = ["//tensorflow/core:lib"],
)

cc_library(
    name = "tensor_coding",
    srcs = ["tensor_coding.cc"],
    hdrs = [
        "tensor_coding.h",
    ],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/protobuf:worker_proto_cc",
    ],
)

cc_library(
    name = "worker_interface",
    hdrs = [
        "worker_interface.h",
    ],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":call_options",
        ":message_wrappers",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core/protobuf:worker_proto_cc",
    ],
)

cc_library(
    name = "error_payloads",
    hdrs = ["error_payloads.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    visibility = ["//visibility:public"],
)

tf_cuda_library(
    name = "worker",
    srcs = ["worker.cc"],
    hdrs = [
        "worker.h",
    ],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":error_payloads",
        ":graph_mgr",
        ":partial_run_mgr",
        ":recent_request_ids",
        ":rendezvous_mgr_interface",
        ":session_mgr",
        ":tensor_coding",
        ":worker_interface",
        ":worker_session",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/profiler/lib:device_profiler_session",
    ],
)

cc_library(
    name = "call_options",
    hdrs = ["call_options.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        "//tensorflow/tsl/distributed_runtime:call_options",
    ],
)

tf_cc_test(
    name = "call_options_test",
    size = "small",
    srcs = ["call_options_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":call_options",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "cancellable_call",
    srcs = ["cancellable_call.cc"],
    hdrs = ["cancellable_call.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":call_options",
        ":worker_cache",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "tensor_coding_test",
    size = "small",
    srcs = ["tensor_coding_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    linkstatic = 1,
    deps = [
        ":tensor_coding",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/framework:tensor_testutil",
        "//tensorflow/core/protobuf:worker_proto_cc",
    ],
)

cc_library(
    name = "worker_cache",
    hdrs = ["worker_cache.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":worker_interface",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/distributed_runtime/coordination:coordination_client",
        "//tensorflow/core/distributed_runtime/eager:eager_client",
    ],
)

cc_library(
    name = "worker_cache_wrapper",
    hdrs = ["worker_cache_wrapper.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [":worker_cache"],
)

cc_library(
    name = "remote_device",
    srcs = ["remote_device.cc"],
    hdrs = ["remote_device.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":worker_cache",
        ":worker_interface",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/protobuf:worker_proto_cc",
    ],
)

cc_library(
    name = "master_interface",
    hdrs = ["master_interface.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":call_options",
        ":message_wrappers",
        ":request_id",
        "//tensorflow/core:lib",
        "//tensorflow/core/protobuf:master_proto_cc",
    ],
)

cc_library(
    name = "master",
    srcs = ["master.cc"],
    hdrs = ["master.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":call_options",
        ":master_env",
        ":master_session",
        ":recent_request_ids",
        ":remote_device",
        ":worker_cache",
        ":worker_interface",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:regexp",
        "//tensorflow/core/protobuf:master_proto_cc",
        "//tensorflow/core/protobuf:worker_proto_cc",
    ],
)

cc_library(
    name = "master_session",
    srcs = ["master_session.cc"],
    hdrs = ["master_session.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":call_options",
        ":master_env",
        ":message_wrappers",
        ":request_id",
        ":scheduler",
        ":worker_cache",
        ":worker_interface",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/debug:debug_graph_utils",
        "//tensorflow/core/protobuf:master_proto_cc",
    ],
)

cc_library(
    name = "local_master",
    srcs = ["local_master.cc"],
    hdrs = ["local_master.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":master",
        ":master_interface",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "rendezvous_mgr_interface",
    srcs = [],
    hdrs = ["rendezvous_mgr_interface.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":worker_env",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "scheduler",
    srcs = ["scheduler.cc"],
    hdrs = ["scheduler.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:tensorflow_opensource",
    ],
)

cc_library(
    name = "base_rendezvous_mgr",
    srcs = ["base_rendezvous_mgr.cc"],
    hdrs = ["base_rendezvous_mgr.h"],
    copts =
        # copybara:uncomment ["-Wthread-safety-analysis"] +
        tf_copts(),
    deps = [
        ":rendezvous_mgr_interface",
        ":worker_env",
        ":worker_session",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/profiler/lib:scoped_memory_debug_annotation",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/hash",
    ],
)

cc_library(
    name = "master_env",
    hdrs = ["master_env.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":worker_cache",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:session_options",
    ],
)

cc_library(
    name = "graph_mgr",
    srcs = ["graph_mgr.cc"],
    hdrs = ["graph_mgr.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":message_wrappers",
        ":rendezvous_mgr_interface",
        ":worker_env",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/debug",
        "//tensorflow/core/profiler/lib:connected_traceme",
        "//tensorflow/core/profiler/lib:traceme_encode",
        "//tensorflow/core/protobuf:worker_proto_cc",
    ],
)

cc_library(
    name = "worker_cache_partial",
    srcs = ["worker_cache_partial.cc"],
    hdrs = ["worker_cache_partial.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":worker_cache",
        ":worker_interface",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/protobuf:worker_proto_cc",
    ],
)

cc_library(
    name = "worker_cache_logger",
    srcs = ["worker_cache_logger.cc"],
    hdrs = ["worker_cache_logger.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "server_lib",
    srcs = ["server_lib.cc"],
    hdrs = ["server_lib.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ],
)

tf_cc_test(
    name = "server_lib_test",
    srcs = ["server_lib_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":server_lib",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "rpc_collective_executor_mgr",
    srcs = ["rpc_collective_executor_mgr.cc"],
    hdrs = ["rpc_collective_executor_mgr.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":collective_param_resolver_distributed",
        ":collective_rma_distributed",
        ":device_resolver_distributed",
        ":worker_cache",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib_internal",
    ],
)

tf_cc_test(
    name = "rpc_collective_executor_mgr_test",
    srcs = ["rpc_collective_executor_mgr_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":collective_param_resolver_distributed",
        ":device_resolver_distributed",
        ":rpc_collective_executor_mgr",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:session_options",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/nccl:collective_communicator",
        "//tensorflow/core/protobuf:worker_proto_cc",
    ],
)

cc_library(
    name = "collective_rma_distributed",
    srcs = ["collective_rma_distributed.cc"],
    hdrs = ["collective_rma_distributed.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":call_options",
        ":cancellable_call",
        ":request_id",
        ":worker_cache",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib_internal",  # protobuf::Any
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/profiler/lib:scoped_memory_debug_annotation",
        "//tensorflow/core/protobuf:worker_proto_cc",
        "@com_google_absl//absl/memory",
    ],
)

tf_cc_test(
    name = "collective_rma_distributed_test",
    size = "small",
    srcs = ["collective_rma_distributed_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    tags = [
        "no_windows",  # TODO(b/187863739): Fix failure and remove.
    ],
    deps = [
        ":collective_rma_distributed",
        ":device_resolver_distributed",
        ":test_utils",
        "//tensorflow/core:core_cpu_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/protobuf:worker_proto_cc",
    ],
)

cc_library(
    name = "collective_param_resolver_distributed",
    srcs = ["collective_param_resolver_distributed.cc"],
    hdrs = ["collective_param_resolver_distributed.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":cancellable_call",
        ":device_resolver_distributed",
        ":worker_cache",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "test_utils",
    srcs = [],
    hdrs = ["test_utils.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":worker_cache",
        ":worker_interface",
        "//tensorflow/core:framework",
    ],
)

tf_cc_test(
    name = "collective_param_resolver_distributed_test",
    size = "small",
    srcs = ["collective_param_resolver_distributed_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":collective_param_resolver_distributed",
        ":device_resolver_distributed",
        ":test_utils",
        "//tensorflow/core:collective_ops_op_lib",
        "//tensorflow/core:core_cpu_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:no_op_op_lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:sendrecv_ops_op_lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/distributed_runtime:worker",
        "//tensorflow/core/distributed_runtime:worker_env",
        "//tensorflow/core/kernels:collective_ops",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_library(
    name = "device_resolver_distributed",
    srcs = ["device_resolver_distributed.cc"],
    hdrs = ["device_resolver_distributed.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":worker_cache",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

tf_cc_test(
    name = "device_resolver_distributed_test",
    size = "small",
    srcs = ["device_resolver_distributed_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":device_resolver_distributed",
        ":test_utils",
        "//tensorflow/core:core_cpu_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "@com_google_absl//absl/memory",
    ],
)

tf_cuda_cc_test(
    name = "master_test",
    size = "medium",
    srcs = [
        "master_test.cc",
    ],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":master",
        ":remote_device",
        ":worker_interface",
        "//tensorflow/core:control_flow_ops_op_lib",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:state_ops_op_lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/distributed_runtime/rpc:grpc_channel",
        "//tensorflow/core/distributed_runtime/rpc:grpc_master_service_impl",
        "//tensorflow/core/distributed_runtime/rpc:grpc_testlib",
        "//tensorflow/core/distributed_runtime/rpc:grpc_util",
        "//tensorflow/core/distributed_runtime/rpc:grpc_worker_cache",
        "//tensorflow/core/kernels:control_flow_ops",
        "//tensorflow/core/kernels:cwise_op",
        "//tensorflow/core/kernels:dense_update_ops",
        "//tensorflow/core/kernels:identity_op",
        "//tensorflow/core/kernels:variable_ops",
        "//tensorflow/core/protobuf:master_proto_cc",
    ] + tf_grpc_cc_dependencies(),
)

tf_cuda_cc_test(
    name = "remote_device_test",
    size = "small",
    srcs = ["remote_device_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    tags = tf_cuda_tests_tags() + [
        "manual",
        "noguitar",  # Memory leak due to b/62910646
        "notap",  # Memory leak due to b/62910646
    ],
    deps = [
        ":master",
        ":remote_device",
        ":worker_interface",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/distributed_runtime/rpc:grpc_channel",
        "//tensorflow/core/distributed_runtime/rpc:grpc_testlib",
        "//tensorflow/core/distributed_runtime/rpc:grpc_util",
        "//tensorflow/core/distributed_runtime/rpc:grpc_worker_cache",
        "//tensorflow/core/protobuf:master_proto_cc",
    ] + tf_grpc_cc_dependencies(),
)

tf_cuda_cc_test(
    name = "rpcbench_test",
    size = "small",
    srcs = ["rpcbench_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    linkstatic = 1,
    tags = tf_cuda_tests_tags(),
    deps = [
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:array_ops_op_lib",
        "//tensorflow/core:bitwise_ops_op_lib",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:functional_ops_op_lib",
        "//tensorflow/core:lib",
        "//tensorflow/core:math_ops_op_lib",
        "//tensorflow/core:nn_ops_op_lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensorflow",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/distributed_runtime:server_lib",
        "//tensorflow/core/distributed_runtime/rpc:grpc_server_lib",
        "//tensorflow/core/distributed_runtime/rpc:grpc_session",
        "//tensorflow/core/kernels:aggregate_ops",
        "//tensorflow/core/kernels:array",
    ],
)

cc_library(
    name = "request_id",
    srcs = ["request_id.cc"],
    hdrs = ["request_id.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_cc_test(
    name = "request_id_test",
    size = "small",
    srcs = ["request_id_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":request_id",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "recent_request_ids",
    srcs = ["recent_request_ids.cc"],
    hdrs = ["recent_request_ids.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":message_wrappers",
        "//tensorflow/core:lib",
        "//tensorflow/core/protobuf:worker_proto_cc",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

tf_cc_test(
    name = "recent_request_ids_test",
    size = "small",
    srcs = ["recent_request_ids_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":recent_request_ids",
        ":request_id",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/protobuf:worker_proto_cc",
    ],
)

filegroup(
    name = "pywrap_required_hdrs",
    srcs = [
        "call_options.h",
        "message_wrappers.h",
        "rendezvous_mgr_interface.h",
        "server_lib.h",
        "worker_cache.h",
        "worker_env.h",
        "worker_interface.h",
    ],
    visibility = [
        "//tensorflow/core:__pkg__",
        "//tensorflow/core/function/runtime_client:__pkg__",
        "//tensorflow/python:__subpackages__",
    ],
)

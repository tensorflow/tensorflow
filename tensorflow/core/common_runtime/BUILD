load(
    "//tensorflow:tensorflow.bzl",
    "cc_header_only_library",
    "tf_cc_test",
    "tf_cc_test_mkl",
    "tf_cc_tests",
    "tf_copts",
    "tf_cuda_library",
    "tf_openmp_copts",
)

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "tf_cc_test_gpu")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "tf_cc_tests_gpu")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "tf_cuda_cc_test")

# For platform specific build config
load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_additional_core_deps",
    "tf_kernel_tests_linkstatic",
    "tf_protos_all",
    "tf_protos_grappler",
)
load(
    "//tensorflow/core/platform:rules_cc.bzl",
    "cc_library",
)
load(
    "//tensorflow/core/platform:build_config_root.bzl",
    "if_static",
    "tf_cuda_tests_tags",
)
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")
load(
    "//third_party/mkl:build_defs.bzl",
    "if_mkl",
    "mkl_deps",
)

package(
    default_visibility = [
        "//tensorflow:internal",
        "//tensorflow_models:__subpackages__",
    ],
    licenses = ["notice"],  # Apache 2.0
)

tf_cuda_library(
    name = "core_cpu",
    hdrs = [
        "composite_device.h",
        "device.h",
        "device_factory.h",
        "function.h",
        "function_optimization_registry.h",
        "optimization_registry.h",
        "shape_refiner.h",
        "//tensorflow/core/graph:core_cpu_headers",
        "//tensorflow/core/public:session.h",
        "//tensorflow/core/public:session_options.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":core_cpu_internal",
    ],
)

cc_header_only_library(
    name = "core_cpu_headers_lib",
    visibility = ["//visibility:public"],
    deps = [
        ":core_cpu_lib",
    ],
)

cc_library(
    name = "core",
    visibility = ["//visibility:public"],
    deps = [
        ":core_cpu",
        "//tensorflow/core/common_runtime/gpu:gpu_runtime",
        "//tensorflow/core/common_runtime/sycl:sycl_runtime",
    ],
)

filegroup(
    name = "testlib_srcs",
    srcs = [
        "function_testlib.cc",
        "kernel_benchmark_testlib.cc",
    ],
)

filegroup(
    name = "testlib_headers",
    srcs = [
        "function_testlib.h",
        "kernel_benchmark_testlib.h",
        "test_collective_executor_mgr.h",
    ],
)

cc_library(
    name = "testlib_ops",
    testonly = 1,
    srcs = ["testlib_ops.cc"],
    linkstatic = 1,  # Seems to be needed since alwayslink is broken in bazel
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
    alwayslink = 1,
)

# -----------------------------------------------------------------------------
# Public Android targets

# Sources required to build the TensorFlow framework with runtime on
# mobile platforms without granular targets. It is assumed that the source
# files in tensorflow/core:mobile_srcs_no_runtime have been compiled
# separately and are linked in as a dependency.
filegroup(
    name = "mobile_srcs_only_runtime",
    srcs = [
    ] + glob(
        [
            "**/*.cc",
            "**/*.h",
        ],
        exclude = [
            "**/*test.*",
            "**/*testutil*",
            "**/*testlib*",
            "**/*main.cc",
            "gpu/**/*",
            "gpu_device_factory.*",
        ],
    ),
)

filegroup(
    name = "core_cpu_base_headers",
    srcs = [
        "composite_device.h",
        "device.h",
        "device_factory.h",
        "device_mgr.h",
        "device_set.h",
        "eval_const_tensor.h",
        "function.h",
        "graph_runner.h",
        "metrics.h",
        "process_function_library_runtime.h",
        "scoped_allocator.h",
        "scoped_allocator_mgr.h",
        "shape_refiner.h",
        "//tensorflow/core/framework:versions.h",
        "//tensorflow/core/graph:graph_headers",
    ],
)

tf_cuda_library(
    name = "core_cpu_base_no_ops",
    srcs = [
        "eval_const_tensor.cc",
        "graph_optimizer.h",
        "shape_refiner.cc",
        "//tensorflow/core/graph:core_cpu_base_no_ops_srcs",
        "//tensorflow/core/public:session_options.h",
        "//tensorflow/core/public:version.h",
    ],
    hdrs = [
        ":core_cpu_base_headers",
        "//tensorflow/core/public:session.h",
    ],
    copts = tf_copts(),
    deps = [
        ":scoped_allocator",
        "//tensorflow/core:graph",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/container:flat_hash_set",
        "//third_party/eigen3",
    ] + if_static([
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
    ]),
)

filegroup(
    name = "core_cpu_lib_headers",
    srcs = [
        ":core_cpu_base_headers",
        "allocator_retry.h",
        "shared_counter.h",
        "base_collective_executor.h",
        "bfc_allocator.h",
        "hierarchical_tree_broadcaster.h",
        "buf_rendezvous.h",
        "build_graph_options.h",
        "collective_executor_mgr.h",
        "collective_param_resolver_local.h",
        "collective_rma_local.h",
        "collective_util.h",
        "colocation_graph.h",
        "constant_folding.h",
        "copy_tensor.h",
        "costmodel_manager.h",
        "placer_inspection_required_ops_utils.h",
        "debugger_state_interface.h",
        "device_resolver_local.h",
        "dma_helper.h",
        "executor.h",
        "executor_factory.h",
        "function_optimization_registry.h",
        "graph_optimizer.h",
        "input_colocation_exemption_registry.h",
        "isolate_placer_inspection_required_ops_pass.h",
        "local_device.h",
        "local_executor_params.h",
        "lower_function_call_op.h",
        "lower_if_op.h",
        "lower_case_op.h",
        "lower_functional_ops.h",
        "lower_while_op.h",
        "memory_types.h",
        "mkl_cpu_allocator.h",
        "optimization_registry.h",
        "partitioning_utils.h",
        "placer.h",
        "process_util.h",
        "inspecting_placer.h",
        "profile_handler.h",
        "renamed_device.h",
        "rendezvous_mgr.h",
        "rendezvous_util.h",
        "replicate_per_replica_nodes.h",
        "ring_reducer.h",
        "ring_alg.h",
        "ring_gatherer.h",
        "session_factory.h",
        "single_threaded_cpu_device.h",
        "stats_publisher_interface.h",
        "step_stats_collector.h",
        "threadpool_device.h",
        "process_state.h",
        "pool_allocator.h",
        "//tensorflow/core/graph:core_cpu_lib_headers",
    ] + if_mkl(["//tensorflow/core/graph:mkl_graph_util_header"]),
)

cc_library(
    name = "accumulate_n_optimizer",
    srcs = ["accumulate_n_optimizer.cc"],
    copts = tf_copts(),
    deps = [
        ":optimization_registry",
        "//tensorflow/core:graph",
    ],
    alwayslink = 1,
)

cc_library(
    name = "base_collective_executor",
    srcs = ["base_collective_executor.cc"],
    hdrs = ["base_collective_executor.h"],
    copts = tf_copts(),
    deps = [
        ":buf_rendezvous",
        ":copy_tensor",
        ":device_mgr",
        ":dma_helper",
        ":process_util",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/profiler/lib:traceme",
    ],
)

cc_library(
    name = "buf_rendezvous",
    srcs = ["buf_rendezvous.cc"],
    hdrs = ["buf_rendezvous.h"],
    copts = tf_copts(),
    deps = [
        ":device",
        ":device_mgr",
        ":process_util",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "build_graph_options",
    srcs = ["build_graph_options.cc"],
    hdrs = ["build_graph_options.h"],
    copts = tf_copts(),
    deps = [
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "collective_executor_mgr",
    srcs = ["collective_executor_mgr.cc"],
    hdrs = ["collective_executor_mgr.h"],
    copts = tf_copts(),
    deps = [
        ":base_collective_executor",
        ":build_graph_options",
        ":collective_rma_local",
        ":device_mgr",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "collective_util",
    srcs = ["collective_util.cc"],
    hdrs = ["collective_util.h"],
    copts = tf_copts(),
    deps = [
        ":device",
        ":device_mgr",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "copy_tensor",
    srcs = ["copy_tensor.cc"],
    hdrs = ["copy_tensor.h"],
    copts = tf_copts(),
    deps = [
        ":device",
        ":dma_helper",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/profiler/lib:scoped_annotation",
    ],
)

cc_library(
    name = "collective_param_resolver_local",
    srcs = ["collective_param_resolver_local.cc"],
    hdrs = ["collective_param_resolver_local.h"],
    copts = tf_copts(),
    deps = [
        ":device",
        ":device_mgr",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "collective_rma_local",
    srcs = ["collective_rma_local.cc"],
    hdrs = ["collective_rma_local.h"],
    copts = tf_copts(),
    deps = [
        ":buf_rendezvous",
        ":copy_tensor",
        ":device",
        ":device_mgr",
        ":dma_helper",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "costmodel_manager",
    srcs = ["costmodel_manager.cc"],
    hdrs = ["costmodel_manager.h"],
    copts = tf_copts(),
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

cc_library(
    name = "debugger_state_interface",
    srcs = ["debugger_state_interface.cc"],
    hdrs = ["debugger_state_interface.h"],
    copts = tf_copts(),
    deps = [
        ":device",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "device",
    srcs = ["device.cc"],
    hdrs = ["device.h"],
    copts = tf_copts(),
    deps = [
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "device_factory",
    srcs = ["device_factory.cc"],
    hdrs = ["device_factory.h"],
    copts = tf_copts(),
    deps = [
        ":device",
        ":session_options",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

cc_library(
    name = "device_mgr",
    srcs = [
        "device_mgr.cc",
        "dynamic_device_mgr.cc",
    ],
    hdrs = ["device_mgr.h"],
    copts = tf_copts(),
    deps = [
        ":device",
        ":local_device",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "device_resolver_local",
    srcs = ["device_resolver_local.cc"],
    hdrs = ["device_resolver_local.h"],
    copts = tf_copts(),
    deps = [
        ":device_mgr",
        "//tensorflow/core:framework",
    ],
)

cc_library(
    name = "entry",
    hdrs = ["entry.h"],
    copts = tf_copts(),
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "executor",
    srcs = ["executor.cc"],
    hdrs = ["executor.h"],
    copts = tf_copts(),
    deps = [
        ":costmodel_manager",
        ":device",
        ":entry",
        ":executor_factory",
        ":graph_view",
        ":immutable_executor_state",
        ":local_executor_params",
        ":pending_counts",
        ":propagator_state",
        ":renamed_device",
        ":simple_propagator_state",
        ":step_stats_collector",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/profiler/lib:annotated_traceme",
        "//tensorflow/core/profiler/lib:scoped_annotation",
        "//tensorflow/core/profiler/lib:traceme",
        "@com_google_absl//absl/memory",
    ],
    alwayslink = 1,
)

cc_library(
    name = "executor_factory",
    srcs = ["executor_factory.cc"],
    hdrs = ["executor_factory.h"],
    copts = tf_copts(),
    deps = [
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "graph_view",
    srcs = ["graph_view.cc"],
    hdrs = ["graph_view.h"],
    copts = tf_copts(),
    deps = [
        ":device",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "device_set",
    srcs = ["device_set.cc"],
    hdrs = ["device_set.h"],
    copts = tf_copts(),
    deps = [
        ":device",
        ":device_factory",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

cc_library(
    name = "dma_helper",
    hdrs = ["dma_helper.h"],
    copts = tf_copts(),
    deps = ["//tensorflow/core:framework"],
)

cc_library(
    name = "hierarchical_tree_broadcaster",
    srcs = ["hierarchical_tree_broadcaster.cc"],
    hdrs = ["hierarchical_tree_broadcaster.h"],
    copts = tf_copts(),
    deps = [
        ":base_collective_executor",
        ":collective_rma_local",
        ":collective_util",
        ":device_mgr",
        ":dma_helper",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/profiler/lib:traceme",
    ],
    alwayslink = 1,
)

cc_library(
    name = "immutable_executor_state",
    srcs = ["immutable_executor_state.cc"],
    hdrs = ["immutable_executor_state.h"],
    copts = tf_copts(),
    deps = [
        ":graph_view",
        ":local_executor_params",
        ":pending_counts",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "input_colocation_exemption_registry",
    srcs = ["input_colocation_exemption_registry.cc"],
    hdrs = ["input_colocation_exemption_registry.h"],
    copts = tf_copts(),
    deps = [
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "local_device",
    srcs = ["local_device.cc"],
    hdrs = ["local_device.h"],
    copts = tf_copts(),
    deps = [
        ":device",
        ":process_state",
        ":process_util",
        ":session_options",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//third_party/eigen3",
    ],
)

cc_library(
    name = "local_executor_params",
    hdrs = ["local_executor_params.h"],
    copts = tf_copts(),
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "memory_types",
    srcs = ["memory_types.cc"],
    hdrs = ["memory_types.h"],
    copts = tf_copts(),
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

cc_library(
    name = "mkl_cpu_allocator",
    srcs = ["mkl_cpu_allocator.cc"],
    hdrs = ["mkl_cpu_allocator.h"],
    copts = tf_copts(),
    deps = [
        ":bfc_allocator",
        ":pool_allocator",
        "//tensorflow/core:lib",
    ] + mkl_deps(),
)

cc_library(
    name = "optimization_registry",
    srcs = ["optimization_registry.cc"],
    hdrs = ["optimization_registry.h"],
    copts = tf_copts(),
    deps = [
        ":device_set",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "parallel_concat_optimizer",
    srcs = ["parallel_concat_optimizer.cc"],
    copts = tf_copts(),
    deps = [
        ":optimization_registry",
        "//tensorflow/core:graph",
    ],
    alwayslink = 1,
)

cc_library(
    name = "partitioning_utils",
    srcs = ["partitioning_utils.cc"],
    hdrs = ["partitioning_utils.h"],
    copts = tf_copts(),
    deps = [
        ":device_set",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "pending_counts",
    hdrs = ["pending_counts.h"],
    copts = tf_copts(),
    deps = [
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "pool_allocator",
    srcs = ["pool_allocator.cc"],
    hdrs = ["pool_allocator.h"],
    copts = tf_copts(),
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

cc_library(
    name = "process_state",
    srcs = ["process_state.cc"],
    hdrs = ["process_state.h"],
    copts = tf_copts(),
    deps = [
        ":bfc_allocator",
        ":pool_allocator",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/util:env_var",
        "@com_google_absl//absl/base",
    ],
)

cc_library(
    name = "process_util",
    srcs = ["process_util.cc"],
    hdrs = ["process_util.h"],
    copts = tf_copts() + tf_openmp_copts(),
    deps = [
        ":session_options",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

cc_library(
    name = "profile_handler",
    hdrs = ["profile_handler.h"],
    copts = tf_copts(),
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "propagator_debug_utils",
    srcs = ["propagator_debug_utils.cc"],
    hdrs = ["propagator_debug_utils.h"],
    copts = tf_copts(),
    deps = [
        ":entry",
        ":graph_view",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "propagator_state",
    srcs = ["propagator_state.cc"],
    hdrs = ["propagator_state.h"],
    copts = tf_copts(),
    deps = [
        ":entry",
        ":graph_view",
        ":immutable_executor_state",
        ":pending_counts",
        ":propagator_debug_utils",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/profiler/lib:traceme",
    ],
)

cc_library(
    name = "renamed_device",
    srcs = ["renamed_device.cc"],
    hdrs = ["renamed_device.h"],
    copts = tf_copts(),
    deps = [
        ":device",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//third_party/eigen3",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "rendezvous_mgr",
    srcs = ["rendezvous_mgr.cc"],
    hdrs = ["rendezvous_mgr.h"],
    copts = tf_copts(),
    deps = [
        ":copy_tensor",
        ":device",
        ":device_mgr",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "ring_alg",
    srcs = ["ring_alg.cc"],
    hdrs = ["ring_alg.h"],
    copts = tf_copts(),
    deps = [
        ":base_collective_executor",
        ":collective_rma_local",
        ":collective_util",
        ":copy_tensor",
        ":device",
        ":device_mgr",
        ":dma_helper",
        ":process_util",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "ring_gatherer",
    srcs = ["ring_gatherer.cc"],
    hdrs = ["ring_gatherer.h"],
    copts = tf_copts(),
    deps = [
        ":base_collective_executor",
        ":collective_rma_local",
        ":collective_util",
        ":copy_tensor",
        ":device",
        ":device_mgr",
        ":dma_helper",
        ":process_util",
        ":ring_alg",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/profiler/lib:traceme",
    ],
    alwayslink = 1,
)

cc_library(
    name = "ring_reducer",
    srcs = ["ring_reducer.cc"],
    hdrs = ["ring_reducer.h"],
    copts = tf_copts(),
    deps = [
        ":base_collective_executor",
        ":collective_rma_local",
        ":collective_util",
        ":copy_tensor",
        ":device",
        ":device_mgr",
        ":dma_helper",
        ":process_util",
        ":ring_alg",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/profiler/lib:traceme",
    ],
    alwayslink = 1,
)

cc_library(
    name = "rendezvous_util",
    srcs = ["rendezvous_util.cc"],
    hdrs = ["rendezvous_util.h"],
    copts = tf_copts(),
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "replicate_per_replica_nodes",
    srcs = ["replicate_per_replica_nodes.cc"],
    hdrs = ["replicate_per_replica_nodes.h"],
    copts = tf_copts(),
    deps = [
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "scoped_allocator",
    srcs = [
        "scoped_allocator.cc",
        "scoped_allocator_mgr.cc",
    ],
    hdrs = [
        "scoped_allocator.h",
        "scoped_allocator_mgr.h",
    ],
    copts = tf_copts(),
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "session",
    srcs = ["session.cc"],
    hdrs = ["//tensorflow/core/public:session.h"],
    copts = tf_copts(),
    deps = [
        ":session_factory",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "session_factory",
    srcs = ["session_factory.cc"],
    hdrs = ["session_factory.h"],
    copts = tf_copts(),
    deps = [
        ":session_options",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "session_options",
    srcs = ["session_options.cc"],
    hdrs = [
        "//tensorflow/core/public:session_options.h",
    ],
    copts = tf_copts(),
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "simple_propagator_state",
    srcs = ["simple_propagator_state.cc"],
    hdrs = ["simple_propagator_state.h"],
    copts = tf_copts(),
    deps = [
        ":entry",
        ":graph_view",
        ":immutable_executor_state",
        ":pending_counts",
        ":propagator_debug_utils",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/profiler/lib:traceme",
    ],
)

cc_library(
    name = "single_threaded_cpu_device",
    srcs = ["single_threaded_cpu_device.cc"],
    hdrs = [
        "single_threaded_cpu_device.h",
    ],
    copts = tf_copts(),
    deps = [
        ":device",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//third_party/eigen3",
    ],
)

cc_library(
    name = "session_state",
    srcs = ["session_state.cc"],
    hdrs = ["//tensorflow/core/framework:session_state.h"],
    copts = tf_copts(),
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "stats_publisher_interface",
    srcs = ["stats_publisher_interface.cc"],
    hdrs = ["stats_publisher_interface.h"],
    copts = tf_copts(),
    deps = [
        ":build_graph_options",
        ":profile_handler",
        ":session_options",
        "//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "step_stats_collector",
    srcs = ["step_stats_collector.cc"],
    hdrs = ["step_stats_collector.h"],
    copts = tf_copts(),
    deps = [
        ":costmodel_manager",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "threadpool_device",
    srcs = ["threadpool_device.cc"],
    hdrs = ["threadpool_device.h"],
    copts = tf_copts() + tf_openmp_copts(),
    deps = [
        ":device_factory",
        ":local_device",
        ":scoped_allocator",
        ":session_options",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "threadpool_device_factory",
    srcs = ["threadpool_device_factory.cc"],
    copts = tf_copts(),
    deps = [
        ":device_factory",
        ":process_state",
        ":session_options",
        ":threadpool_device",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/memory",
    ],
    alwayslink = 1,
)

tf_cuda_library(
    name = "core_cpu_rump_impl",
    srcs = [
        "colocation_graph.cc",
        "composite_device.cc",
        "constant_folding.cc",
        "function.cc",
        "function_optimization_registry.cc",
        "graph_optimizer.cc",
        "graph_runner.cc",
        "inspecting_placer.cc",
        "isolate_placer_inspection_required_ops_pass.cc",
        "lower_case_op.cc",
        "lower_function_call_op.cc",
        "lower_functional_ops.cc",
        "lower_if_op.cc",
        "lower_while_op.cc",
        "placer.cc",
        "placer_inspection_required_ops_utils.cc",
        "placer_inspection_required_ops_utils.h",
        "process_function_library_runtime.cc",
        "//tensorflow/core/graph:core_cpu_impl_srcs",
    ],
    hdrs = [":core_cpu_lib_headers"],
    copts = tf_copts(),
    deps = [
        ":device",
        ":entry",
        ":executor",
        ":executor_factory",
        ":graph_view",
        ":local_executor_params",
        ":immutable_executor_state",
        ":input_colocation_exemption_registry",
        ":pending_counts",
        ":propagator_debug_utils",
        ":propagator_state",
        ":session_options",
        ":simple_propagator_state",
        ":single_threaded_cpu_device",
        "//tensorflow/core:graph",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:variant",
        "//tensorflow/core/public:version",
        "//tensorflow/core/grappler/utils:functions",
        "//tensorflow/core/profiler/lib:annotated_traceme",
        "//tensorflow/core/profiler/lib:scoped_annotation",
        "//tensorflow/core/profiler/lib:traceme",
    ] + mkl_deps(),
    alwayslink = 1,
)

tf_cuda_library(
    name = "core_cpu_impl",
    hdrs = [":core_cpu_lib_headers"],
    copts = tf_copts(),
    deps = [
        ":accumulate_n_optimizer",
        ":base_collective_executor",
        ":bfc_allocator",
        ":buf_rendezvous",
        ":build_graph_options",
        ":collective_executor_mgr",
        ":collective_param_resolver_local",
        ":collective_rma_local",
        ":collective_util",
        ":copy_tensor",
        ":core_cpu_rump_impl",
        ":costmodel_manager",
        ":debugger_state_interface",
        ":device",
        ":device_factory",
        ":device_mgr",
        ":device_resolver_local",
        ":device_set",
        ":entry",
        ":graph_view",
        ":hierarchical_tree_broadcaster",
        ":input_colocation_exemption_registry",
        ":local_device",
        ":memory_types",
        ":mkl_cpu_allocator",
        ":optimization_registry",
        ":parallel_concat_optimizer",
        ":partitioning_utils",
        ":pending_counts",
        ":pool_allocator",
        ":process_state",
        ":process_util",
        ":profile_handler",
        ":renamed_device",
        ":rendezvous_mgr",
        ":rendezvous_util",
        ":replicate_per_replica_nodes",
        ":ring_alg",
        ":ring_gatherer",
        ":ring_reducer",
        ":session",
        ":session_factory",
        ":session_options",
        ":session_state",
        ":single_threaded_cpu_device",
        ":stats_publisher_interface",
        ":step_stats_collector",
        ":threadpool_device",
        ":threadpool_device_factory",
    ],
)

tf_cuda_library(
    name = "core_cpu_lib",
    hdrs = [":core_cpu_lib_headers"],
    deps = [
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core/grappler:grappler_item",
    ] + if_static([":core_cpu_impl"]) + tf_protos_all() + tf_protos_grappler(),
)

tf_cuda_library(
    name = "core_cpu_lib_no_ops",
    hdrs = [":core_cpu_lib_headers"],
    deps = [
        ":core_cpu_base_no_ops",
        "//tensorflow/core/grappler:grappler_item",
    ] + tf_protos_all() + tf_protos_grappler(),
)

tf_cuda_library(
    name = "core_cpu_internal",
    srcs = [
        "graph_execution_state.cc",
    ],
    hdrs = [
        "graph_execution_state.h",
        ":core_cpu_lib_headers",
    ],
    copts = tf_copts(),
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler/clusters:utils",
        "//tensorflow/core/grappler/clusters:virtual_cluster",
        "//tensorflow/core/grappler/optimizers:meta_optimizer",
        "//third_party/eigen3",
    ] + mkl_deps() + tf_additional_core_deps() + if_static([
        ":core_cpu_impl",
        "//tensorflow/core:function_ops_op_lib",
        "//tensorflow/core:functional_grad",
        "//tensorflow/core:functional_ops_op_lib",
        "//tensorflow/core/kernels:required",
    ]),
    alwayslink = 1,
)

# This is redundant with the "core_cpu_*" targets above. It's useful for
# applications that want to depend on a minimal subset of TensorFlow (e.g. XLA).
cc_library(
    name = "bfc_allocator",
    srcs = [
        "allocator_retry.cc",
        "allocator_retry.h",
        "bfc_allocator.cc",
    ],
    hdrs = ["bfc_allocator.h"],
    features = ["parse_headers"],
    visibility = ["//visibility:public"],
    deps = [
        ":shared_counter",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/framework:allocator",
        "//tensorflow/core/profiler/lib:traceme",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "shared_counter",
    hdrs = ["shared_counter.h"],
    features = ["parse_headers"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:lib",
    ],
)

tf_cuda_library(
    name = "direct_session_internal",
    srcs = ["direct_session.cc"],
    hdrs = [
        "direct_session.h",
    ],
    copts = tf_copts(),
    deps = [
        ":core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_experimental",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/debug:debug_graph_utils",
        "//tensorflow/core/kernels:function_ops",
        "//tensorflow/core/profiler/lib:profiler_backends",
        "//tensorflow/core/profiler/lib:profiler_session",
        "//tensorflow/core/profiler/lib:traceme",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
    alwayslink = 1,
)

filegroup(
    name = "gpu_runtime_headers",
    srcs = [
        "gpu_device_context.h",
    ],
)

# -----------------------------------------------------------------------------
# Tests

tf_cc_test(
    name = "placer_test",
    size = "small",
    srcs = [
        "placer_test.cc",
    ],
    linkopts = select({
        "//tensorflow:macos": ["-headerpad_max_install_names"],
        "//conditions:default": [],
    }),
    linkstatic = tf_kernel_tests_linkstatic(),
    tags = ["no_windows"],
    deps = [
        ":core",
        ":core_cpu",
        ":core_cpu_internal",
        ":direct_session_internal",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:cc_ops_internal",
        "//tensorflow/cc:function_ops",
        "//tensorflow/cc:ops",
        "//tensorflow/cc:scope",
        "//tensorflow/cc:sendrecv_ops",
        "//tensorflow/cc:while_loop",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:ops",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels:ops_util",
        "//tensorflow/core/platform:regexp",
        "//tensorflow/core/util:protos_test_cc",
        "//third_party/eigen3",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "composite_device_test",
    size = "small",
    srcs = [
        "composite_device_test.cc",
    ],
    linkstatic = tf_kernel_tests_linkstatic(),
    deps = [
        ":core_cpu",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_tests(
    name = "core_higher_level_tests",
    size = "small",
    srcs = [
        "buf_rendezvous_test.cc",
        "collective_executor_mgr_test.cc",
        "collective_rma_local_test.cc",
        "device_mgr_test.cc",
        "device_resolver_local_test.cc",
        "device_set_test.cc",
        "dynamic_device_mgr_test.cc",
        "function_optimization_registration_test.cc",
        "function_optimization_registry_no_pass_test.cc",
        "function_optimization_registry_pass_failure_test.cc",
        "function_optimization_registry_test.cc",
        "isolate_placer_inspection_required_ops_pass_test.cc",
        "optimization_registry_test.cc",
        "pending_counts_test.cc",
        "placer_inspection_required_ops_utils_test.cc",
        "session_test.cc",
        "threadpool_device_test.cc",
    ],
    create_named_test_suite = True,
    linkopts = select({
        "//tensorflow:macos": ["-headerpad_max_install_names"],
        "//conditions:default": [],
    }),
    linkstatic = tf_kernel_tests_linkstatic(),
    deps = [
        ":core",
        ":core_cpu",
        ":core_cpu_internal",
        ":direct_session_internal",
        ":pending_counts",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:cc_ops_internal",
        "//tensorflow/cc:function_ops",
        "//tensorflow/cc:ops",
        "//tensorflow/cc:scope",
        "//tensorflow/cc:sendrecv_ops",
        "//tensorflow/cc:while_loop",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:ops",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels:ops_util",
        "//tensorflow/core/platform:regexp",
        "//tensorflow/core/util:protos_test_cc",
        "//third_party/eigen3",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_tests(
    name = "higher_level_tests_needing_kernels",
    size = "small",
    srcs = [
        "collective_param_resolver_local_test.cc",
    ],
    linkopts = select({
        "//tensorflow:macos": ["-headerpad_max_install_names"],
        "//conditions:default": [],
    }),
    linkstatic = tf_kernel_tests_linkstatic(),
    deps = [
        ":core",
        ":core_cpu",
        ":core_cpu_internal",
        ":direct_session_internal",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:cc_ops_internal",
        "//tensorflow/cc:scope",
        "//tensorflow/cc:sendrecv_ops",
        "//tensorflow/core:all_kernels",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:ops",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels:ops_util",
        "//tensorflow/core/util:protos_test_cc",
        "//third_party/eigen3",
    ],
)

tf_cc_tests_gpu(
    name = "ring_reducer_test",
    size = "medium",
    srcs = [
        "ring_reducer_test.cc",
    ],
    linkstatic = tf_kernel_tests_linkstatic(),
    tags = ["no_cuda_on_cpu_tap"],
    deps = [
        ":core",
        ":core_cpu",
        ":core_cpu_internal",
        "//tensorflow/core:all_kernels",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:gpu_runtime",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:ops",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/util:protos_test_cc",
        "@com_google_absl//absl/memory",
    ],
)

tf_cc_tests_gpu(
    name = "ring_gatherer_test",
    size = "medium",
    srcs = [
        "ring_gatherer_test.cc",
    ],
    linkstatic = tf_kernel_tests_linkstatic(),
    tags = ["no_cuda_on_cpu_tap"],
    deps = [
        ":core",
        ":core_cpu",
        ":core_cpu_internal",
        "//tensorflow/core:all_kernels",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:ops",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/common_runtime/gpu:gpu_runtime",
        "//tensorflow/core/util:protos_test_cc",
        "@com_google_absl//absl/memory",
    ],
)

tf_cc_tests_gpu(
    name = "hierarchical_tree_broadcaster_test",
    size = "medium",
    srcs = [
        "hierarchical_tree_broadcaster_test.cc",
    ],
    linkstatic = tf_kernel_tests_linkstatic(),
    tags = ["no_cuda_on_cpu_tap"],
    deps = [
        ":core",
        ":core_cpu",
        ":core_cpu_internal",
        "//tensorflow/core:all_kernels",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:ops",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/common_runtime/gpu:gpu_runtime",
        "//tensorflow/core/util:protos_test_cc",
        "@com_google_absl//absl/memory",
    ],
)

tf_cc_test_mkl(
    name = "mkl_runtime_tests",
    size = "small",
    srcs = [
        "mkl_cpu_allocator_test.cc",
        "mkl_threadpool_device_test.cc",
    ],
    linkstatic = 1,
    deps = [
        ":core",
        ":core_cpu",
        ":core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test_gpu(
    name = "memory_types_test",
    size = "small",
    srcs = ["memory_types_test.cc"],
    linkstatic = tf_kernel_tests_linkstatic(),
    tags = tf_cuda_tests_tags(),
    deps = [
        ":memory_types",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:ops",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/common_runtime/gpu:gpu_runtime",
        "//tensorflow/core/kernels:cast_op",
        "//third_party/eigen3",
    ],
)

tf_cc_test(
    name = "constant_folding_test",
    size = "small",
    srcs = ["constant_folding_test.cc"],
    linkstatic = tf_kernel_tests_linkstatic(),
    tags = tf_cuda_tests_tags(),
    deps = [
        ":core",
        ":core_cpu",
        ":core_cpu_internal",
        ":direct_session_internal",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:cc_ops_internal",
        "//tensorflow/cc:sendrecv_ops",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:ops",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/common_runtime/gpu:gpu_runtime",
        "//tensorflow/core/kernels:bcast_ops",
        "//tensorflow/core/kernels:cast_op",
        "//tensorflow/core/kernels:concat_op",
        "//tensorflow/core/kernels:cwise_op",
        "//tensorflow/core/kernels:identity_op",
        "//tensorflow/core/kernels:immutable_constant_op",
        "//tensorflow/core/kernels:matmul_op",
        "//tensorflow/core/kernels:topk_op",
        "//third_party/eigen3",
    ],
)

tf_cc_test(
    name = "shape_refiner_test",
    size = "small",
    srcs = [
        "shape_refiner_test.cc",
    ],
    linkstatic = tf_kernel_tests_linkstatic(),
    deps = [
        ":core",
        ":core_cpu",
        ":core_cpu_internal",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:resource_variable_ops",
        "//tensorflow/cc:scope",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:ops",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels:array",
        "//tensorflow/core/kernels:math",
        "//tensorflow/core/kernels:resource_variable_ops",
        "//third_party/eigen3",
    ],
)

tf_cuda_cc_test(
    name = "process_function_library_runtime_test",
    size = "small",
    srcs = ["process_function_library_runtime_test.cc"],
    linkstatic = tf_kernel_tests_linkstatic(),
    tags = ["no_rocm"],
    deps = [
        ":core_cpu",
        ":core_cpu_internal",
        "//tensorflow/cc:function_ops",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels:cast_op",
        "//tensorflow/core/kernels:cwise_op",
        "//tensorflow/core/kernels:function_ops",
        "//tensorflow/core/kernels:resource_variable_ops",
    ],
)

tf_cc_test(
    name = "process_util_test",
    size = "small",
    srcs = ["process_util_test.cc"],
    linkstatic = tf_kernel_tests_linkstatic(),
    deps = [
        ":process_util",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "rendezvous_util_test",
    size = "small",
    srcs = ["rendezvous_util_test.cc"],
    linkstatic = tf_kernel_tests_linkstatic(),
    deps = [
        ":rendezvous_util",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "replicate_per_replica_nodes_test",
    size = "small",
    srcs = ["replicate_per_replica_nodes_test.cc"],
    linkstatic = tf_kernel_tests_linkstatic(),
    deps = [
        ":replicate_per_replica_nodes",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:cc_ops_internal",
        "//tensorflow/cc:function_ops",
        "//tensorflow/cc:ops",
        "//tensorflow/cc:resource_variable_ops",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "partitioning_utils_test",
    size = "small",
    srcs = ["partitioning_utils_test.cc"],
    deps = [
        ":core_cpu",
        ":core_cpu_internal",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:cc_ops_internal",
        "//tensorflow/cc:function_ops",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:ops",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels:function_ops",
        "//tensorflow/core/kernels:identity_op",
    ],
)

tf_cuda_cc_test(
    name = "direct_session_test",
    size = "small",
    srcs = ["direct_session_test.cc"],
    args = [] + if_cuda(["--heap_check=local"]),  # The GPU tracer leaks memory
    linkstatic = tf_kernel_tests_linkstatic(),
    deps = [
        ":core_cpu",
        ":core_cpu_internal",
        ":direct_session_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:ops",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "//third_party/eigen3",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core/kernels:collective_ops",
        "//tensorflow/core/kernels:control_flow_ops",
        "//tensorflow/core/kernels:cwise_op",
        "//tensorflow/core/kernels:dense_update_ops",
        "//tensorflow/core/kernels:fifo_queue_op",
        "//tensorflow/core/kernels:function_ops",
        "//tensorflow/core/kernels:identity_n_op",
        "//tensorflow/core/kernels:identity_op",
        "//tensorflow/core/kernels:matmul_op",
        "//tensorflow/core/kernels:ops_util",
        "//tensorflow/core/kernels:queue_ops",
        "//tensorflow/core/kernels:session_ops",
        "//tensorflow/core/kernels:variable_ops",
        "//tensorflow/core/kernels/data:single_threaded_executor",
    ] + if_cuda(["//tensorflow/core/common_runtime/gpu:cuda"]),
)

# This is identical to :common_runtime_direct_session_test with the addition of
# a dependency on alwayslink target //third_party/tensorflow/core/debug, which
# enables support for TensorFlow Debugger (tfdbg).
tf_cc_test(
    name = "direct_session_with_debug_test",
    size = "small",
    srcs = ["direct_session_test.cc"],
    linkstatic = tf_kernel_tests_linkstatic(),
    deps = [
        ":core",
        ":core_cpu",
        ":core_cpu_internal",
        ":direct_session_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:ops",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "@com_google_absl//absl/strings",
        "//third_party/eigen3",
        "@com_google_absl//absl/memory",
        "//tensorflow/cc:cc_ops",
        # Link with support for TensorFlow Debugger (tfdbg).
        "//tensorflow/core/debug",
        "//tensorflow/core/kernels:collective_ops",
        "//tensorflow/core/kernels:control_flow_ops",
        "//tensorflow/core/kernels:cwise_op",
        "//tensorflow/core/kernels:dense_update_ops",
        "//tensorflow/core/kernels:fifo_queue_op",
        "//tensorflow/core/kernels:function_ops",
        "//tensorflow/core/kernels:identity_op",
        "//tensorflow/core/kernels:identity_n_op",
        "//tensorflow/core/kernels:matmul_op",
        "//tensorflow/core/kernels:ops_util",
        "//tensorflow/core/kernels:queue_ops",
        "//tensorflow/core/kernels:session_ops",
        "//tensorflow/core/kernels:variable_ops",
    ],
)

tf_cc_test(
    name = "direct_session_with_tracking_alloc_test",
    size = "small",
    srcs = ["direct_session_with_tracking_alloc_test.cc"],
    args = ["--heap_check=local"],  # The GPU tracer leaks memory
    linkstatic = tf_kernel_tests_linkstatic(),
    tags = ["no_gpu"],
    deps = [
        ":core",
        ":core_cpu",
        ":core_cpu_internal",
        ":direct_session_internal",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:ops",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels:cwise_op",
        "//tensorflow/core/kernels:dense_update_ops",
        "//tensorflow/core/kernels:fifo_queue_op",
        "//tensorflow/core/kernels:identity_op",
        "//tensorflow/core/kernels:matmul_op",
        "//tensorflow/core/kernels:ops_util",
        "//tensorflow/core/kernels:queue_ops",
        "//tensorflow/core/kernels:variable_ops",
        "//third_party/eigen3",
    ],
)

tf_cc_test(
    name = "graph_runner_test",
    size = "small",
    srcs = ["graph_runner_test.cc"],
    linkstatic = tf_kernel_tests_linkstatic(),
    deps = [
        "//tensorflow/core:array_ops_op_lib",
        ":core",
        ":core_cpu",
        ":core_cpu_internal",
        ":direct_session_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:ops",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//third_party/eigen3",
        "//tensorflow/c/kernels:bitcast_op_lib",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:scope",
        "//tensorflow/core/kernels:cwise_op",
    ] + if_mkl([":mkl_array_ops_op_lib"]),
)

tf_cc_test(
    name = "executor_test",
    size = "small",
    srcs = ["executor_test.cc"],
    linkstatic = tf_kernel_tests_linkstatic(),
    deps = [
        ":core",
        ":core_cpu",
        ":core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels:array",
        "//tensorflow/core/kernels:control_flow_ops",
        "//tensorflow/core/kernels:math",
        "//tensorflow/core/kernels:random_ops",
        "//tensorflow/core/kernels:state",
    ],
)

tf_cc_test(
    name = "function_test",
    size = "small",
    srcs = ["function_test.cc"],
    linkstatic = tf_kernel_tests_linkstatic(),
    tags = [
        "manual",
        "no_oss",
    ],
    deps = [
        ":core",
        ":core_cpu",
        ":core_cpu_internal",
        ":direct_session_internal",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:cc_ops_internal",
        "//tensorflow/cc:function_ops",
        "//tensorflow/cc:functional_ops",
        "//tensorflow/cc:sendrecv_ops",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:ops",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels:cast_op",
        "//tensorflow/core/kernels:cwise_op",
        "//tensorflow/core/kernels:function_ops",
        "//tensorflow/core/kernels:matmul_op",
        "//tensorflow/core/kernels:partitioned_function_ops",
        "//tensorflow/core/kernels:random_ops",
        "//tensorflow/core/kernels:shape_ops",
        "//third_party/eigen3",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "function_threadpool_test",
    size = "small",
    srcs = ["function_threadpool_test.cc"],
    linkstatic = tf_kernel_tests_linkstatic(),
    deps = [
        ":core",
        ":core_cpu",
        ":core_cpu_internal",
        ":direct_session_internal",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:cc_ops_internal",
        "//tensorflow/cc:function_ops",
        "//tensorflow/cc:functional_ops",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:ops",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels:cast_op",
        "//tensorflow/core/kernels:cwise_op",
        "//tensorflow/core/kernels:function_ops",
        "//tensorflow/core/kernels:matmul_op",
        "//tensorflow/core/kernels:random_ops",
        "//tensorflow/core/kernels:shape_ops",
        "//third_party/eigen3",
    ],
)

tf_cc_test(
    name = "scoped_allocator_mgr_test",
    size = "small",
    srcs = ["scoped_allocator_mgr_test.cc"],
    linkstatic = tf_kernel_tests_linkstatic(),
    deps = [
        ":dma_helper",
        ":scoped_allocator",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "input_colocation_exemption_registry_test",
    size = "small",
    srcs = ["input_colocation_exemption_registry_test.cc"],
    deps = [
        ":input_colocation_exemption_registry",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test(
    name = "lower_function_call_test",
    size = "small",
    srcs = ["lower_function_call_op_test.cc"],
    deps = [
        ":core_cpu",
        ":core_cpu_internal",
        ":direct_session_internal",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:cc_ops_internal",
        "//tensorflow/cc:client_session",
        "//tensorflow/cc:function_ops",
        "//tensorflow/cc:ops",
        "//tensorflow/cc:resource_variable_ops",
        "//tensorflow/core:all_kernels",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test(
    name = "lower_if_op_test",
    size = "small",
    srcs = ["lower_if_op_test.cc"],
    deps = [
        ":core_cpu",
        ":core_cpu_internal",
        ":direct_session_internal",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:cc_ops_internal",
        "//tensorflow/cc:client_session",
        "//tensorflow/cc:function_ops",
        "//tensorflow/cc:ops",
        "//tensorflow/cc:resource_variable_ops",
        "//tensorflow/core:all_kernels",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test(
    name = "lower_case_op_test",
    size = "small",
    srcs = ["lower_case_op_test.cc"],
    deps = [
        ":core_cpu",
        ":core_cpu_internal",
        ":direct_session_internal",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:cc_ops_internal",
        "//tensorflow/cc:client_session",
        "//tensorflow/cc:function_ops",
        "//tensorflow/cc:ops",
        "//tensorflow/cc:resource_variable_ops",
        "//tensorflow/core:all_kernels",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test(
    name = "lower_while_op_test",
    size = "small",
    srcs = ["lower_while_op_test.cc"],
    deps = [
        ":core_cpu",
        ":core_cpu_internal",
        ":direct_session_internal",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:cc_ops_internal",
        "//tensorflow/cc:client_session",
        "//tensorflow/cc:function_ops",
        "//tensorflow/cc:ops",
        "//tensorflow/core:all_kernels",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "@com_google_absl//absl/algorithm:container",
    ],
)

tf_cc_test(
    name = "lower_functional_ops_test",
    size = "small",
    srcs = ["lower_functional_ops_test.cc"],
    deps = [
        ":core_cpu",
        ":core_cpu_internal",
        ":direct_session_internal",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:cc_ops_internal",
        "//tensorflow/cc:client_session",
        "//tensorflow/cc:function_ops",
        "//tensorflow/cc:ops",
        "//tensorflow/core:all_kernels",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

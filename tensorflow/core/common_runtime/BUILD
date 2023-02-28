load(
    "//tensorflow:tensorflow.bzl",
    "if_google",
    "if_libtpu",
    "if_oss",
    "if_zendnn",
    "tf_cc_test",
    "tf_cc_test_mkl",
    "tf_cc_tests",
    "tf_copts",
    "tf_cuda_library",
    "tf_openmp_copts",
    "tf_openmp_lopts",
)
load("//tensorflow:tensorflow.default.bzl", "cc_header_only_library", "filegroup", "tf_cuda_cc_test")

# For platform specific build config
load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_additional_core_deps",
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
    "if_mkl_ml",
)
load(
    "//tensorflow/security/fuzzing:tf_fuzzing.bzl",
    "tf_cc_fuzz_test",
)

default_package_visibility = [
    "//tensorflow:internal",
    "//tensorflow_models:__subpackages__",
]

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = default_package_visibility,
    features = if_google(
        [
            "-layering_check",
            "-parse_headers",
        ],
        ["-layering_check"],
    ),
    licenses = ["notice"],
)

tf_cuda_library(
    name = "core_cpu",
    hdrs = [
        "composite_device.h",
        "device.h",
        "device_factory.h",
        "device_mgr.h",
        "function.h",
        "function_optimization_registry.h",
        "gradients.h",
        "graph_constructor.h",
        "graph_def_builder_util.h",
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
        "//tensorflow/core/common_runtime/pluggable_device:pluggable_device_runtime",
    ] + if_libtpu(["//tensorflow/core/tpu:tpu_runtime"]),
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

cc_library(
    name = "collective_test_util",
    testonly = 1,
    srcs = ["collective_test_util.cc"],
    hdrs = ["collective_test_util.h"],
    copts = tf_copts(),
    deps = [
        ":device_resolver_local",
        ":process_util",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:session_options",
        "//tensorflow/core:testlib",
        "//tensorflow/core/framework:allocator",
        "//tensorflow/core/framework:device_attributes_proto_cc",
        "//tensorflow/core/framework:types_proto_cc",
        "//tensorflow/core/nccl:collective_communicator",
        "//tensorflow/core/platform:refcount",
        "//tensorflow/core/platform:status",
    ],
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
        "//tensorflow/tsl/framework:allocator_retry.h",
        "//tensorflow/tsl/framework:allocator_retry.cc",
        "//tensorflow/tsl/framework:bfc_allocator.h",
        "//tensorflow/tsl/framework:bfc_allocator.cc",
        "//tensorflow/tsl/framework:shared_counter.h",
    ] + glob(
        [
            "**/*.cc",
            "**/*.h",
        ],
        exclude = [
            "**/*fuzz.*",
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
        "function_body.h",
        "function_def_utils.h",
        "function_utils.h",
        "graph_constructor.h",
        "graph_def_builder_util.h",
        "graph_optimizer.h",
        "graph_runner.h",
        "inline_function_utils.h",
        "lower_function_call_inline_policy.h",
        "optimized_function_graph_info.h",
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
        ":stats_publisher_interface",
        "//tensorflow/core:graph",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/container:flat_hash_set",
        "//third_party/eigen3",
    ] + if_static([
        ":graph_constructor",
        ":graph_def_builder_util",
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
        "arg_ret_placement.h",
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
        "all_to_all.h",
        "device_resolver_local.h",
        "dma_helper.h",
        "executor.h",
        "executor_factory.h",
        "function_optimization_registry.h",
        "graph_optimizer.h",
        "gradients.h",
        "input_colocation_exemption_registry.h",
        "int32_fulltype.h",
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
        "mkl_layout_pass.h",
        "mkl_tfconversion_pass.h",
        "node_file_writer.h",
        "optimization_registry.h",
        "partitioning_utils.h",
        "placer.h",
        "process_util.h",
        "inspecting_placer.h",
        "profile_handler.h",
        "quantize_training.h",
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
        "permuter.h",
        "//tensorflow/tsl/framework:allocator_retry.h",
        "//tensorflow/tsl/framework:shared_counter.h",
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
    name = "all_to_all",
    srcs = ["all_to_all.cc"],
    hdrs = ["all_to_all.h"],
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
        "//tensorflow/core/profiler/lib:traceme",
    ],
    alwayslink = 1,
)

cc_library(
    name = "arg_ret_placement",
    srcs = ["arg_ret_placement.cc"],
    hdrs = [
        "arg_ret_placement.h",
    ],
    copts = tf_copts(),
    deps = [
        "//tensorflow/core:graph",
    ],
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
        "//tensorflow/core/profiler/lib:connected_traceme",
        "//tensorflow/core/profiler/lib:scoped_memory_debug_annotation",
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
        ":collective_param_resolver_local",
        ":collective_rma_local",
        ":device_mgr",
        ":device_resolver_local",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/memory",
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
    name = "colocation_graph",
    srcs = [
        "colocation_graph.cc",
        "inspecting_placer.cc",
    ],
    hdrs = [
        "colocation_graph.h",
        "inspecting_placer.h",
    ],
    copts = tf_copts(),
    deps = [
        ":composite_device",
        ":device",
        ":device_set",
        ":function_body",
        ":function_def_utils",
        ":input_colocation_exemption_registry",
        ":partitioning_utils",
        ":placer_inspection_required_ops_utils",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "composite_device",
    srcs = ["composite_device.cc"],
    hdrs = ["composite_device.h"],
    copts = tf_copts(),
    deps = [
        ":device",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "constant_folding",
    srcs = ["constant_folding.cc"],
    hdrs = ["constant_folding.h"],
    copts = tf_copts(),
    deps = [
        ":device",
        ":device_factory",
        ":executor",
        ":function_utils",
        ":graph_constructor",
        ":memory_types",
        ":rendezvous_mgr",
        ":session_options",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
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
    hdrs = ["device.h"],
    copts = tf_copts(),
    deps = [
        "//tensorflow/core:framework_internal",
    ],
)

cc_library(
    name = "device_factory",
    hdrs = ["device_factory.h"],
    copts = tf_copts(),
    deps = [
        "//tensorflow/core:framework_internal",
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
        "//tensorflow/core/framework:device_attributes_proto_cc",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
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
        "//tensorflow/core/activity_watcher",
        "//tensorflow/core/profiler/lib:annotated_traceme",
        "//tensorflow/core/profiler/lib:connected_traceme",
        "//tensorflow/core/profiler/lib:scoped_annotation",
        "//tensorflow/core/profiler/lib:traceme_encode",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:optional",
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
    name = "type_inference",
    srcs = ["type_inference.cc"],
    hdrs = ["type_inference.h"],
    copts = tf_copts(),
    visibility = default_package_visibility,
    deps = [
        ":optimization_registry",
        "//tensorflow/core:framework",
        "//tensorflow/core/framework:attr_value_proto_cc",
        "//tensorflow/core/framework:node_def_proto_cc",
        "//tensorflow/core/framework:tensor_proto_cc",
        "//tensorflow/core/platform:errors",
    ],
)

cc_library(
    name = "single_threaded_executor",
    srcs = ["single_threaded_executor.cc"],
    hdrs = ["single_threaded_executor.h"],
    copts = tf_copts(),
    deps = [
        ":entry",
        ":executor",
        ":local_executor_params",
        "//tensorflow/core:lib",
    ],
    alwayslink = 1,
)

tf_cc_test(
    name = "eval_const_tensor_test",
    size = "small",
    srcs = ["eval_const_tensor_test.cc"],
    deps = [
        ":graph_constructor",
        "//tensorflow/cc:array_ops",
        "//tensorflow/cc:const_op",
        "//tensorflow/cc:function_ops",
        "//tensorflow/cc:math_ops",
        "//tensorflow/cc:ops",
        "//tensorflow/cc:scope",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_types_hdr",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/framework:tensor_testutil",
        "//tensorflow/core/framework:types_proto_cc",
        "//tensorflow/core/framework:versions_proto_cc",
        "//tensorflow/core/kernels:constant_op",
        "//tensorflow/core/kernels:cwise_op",
        "//tensorflow/core/kernels:function_ops",
        "//tensorflow/core/kernels:identity_op",
        "//tensorflow/core/kernels:slice_op",
        "//tensorflow/core/platform:statusor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/meta:type_traits",
    ],
)

tf_cc_test(
    name = "type_inference_test",
    size = "small",
    srcs = ["type_inference_test.cc"],
    deps = [
        ":core_cpu",
        ":core_cpu_internal",
        ":direct_session_internal",
        ":type_inference",
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
    name = "single_threaded_executor_test",
    size = "small",
    srcs = ["single_threaded_executor_test.cc"],
    deps = [
        "//tensorflow/core:bitwise_ops_op_lib",
        "//tensorflow/core:control_flow_ops_op_lib",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:math_ops_op_lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:random_ops_op_lib",
        "//tensorflow/core:spectral_ops_op_lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels:array",
        "//tensorflow/core/kernels:control_flow_ops",
        "//tensorflow/core/kernels:function_ops",
        "//tensorflow/core/kernels:math",
        "//tensorflow/core/kernels:random_ops",
        "//tensorflow/core/kernels:state",
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
    name = "function",
    srcs = [
        "function.cc",
        "process_function_library_runtime.cc",
    ],
    hdrs = [
        "function.h",
        "process_function_library_runtime.h",
    ],
    copts = tf_copts(),
    deps = [
        ":arg_ret_placement",
        ":composite_device",
        ":device",
        ":device_mgr",
        ":device_set",
        ":executor",
        ":executor_factory",
        ":function_body",
        ":function_def_utils",
        ":function_optimization_registry",
        ":function_utils",
        ":gradients",
        ":graph_constructor",
        ":graph_optimizer",
        ":inline_function_utils",
        ":int32_fulltype",
        ":memory_types",
        ":optimization_registry",
        ":optimize_function_graph_utils",
        ":optimized_function_graph_info",
        ":partitioning_utils",
        ":placer",
        ":process_util",
        ":rendezvous_mgr",
        ":rendezvous_util",
        ":replicate_per_replica_nodes",
        ":single_threaded_executor",
        ":stats_publisher_interface",
        ":type_inference",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/profiler/lib:connected_traceme",
        "//tensorflow/core/profiler/lib:traceme",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:variant",
    ],
)

cc_library(
    name = "function_body",
    srcs = ["function_body.cc"],
    hdrs = ["function_body.h"],
    copts = tf_copts(),
    deps = [
        ":device",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "function_optimization_registry",
    srcs = ["function_optimization_registry.cc"],
    hdrs = ["function_optimization_registry.h"],
    deps = [
        ":device_set",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "function_def_utils",
    srcs = ["function_def_utils.cc"],
    hdrs = ["function_def_utils.h"],
    copts = tf_copts(),
    deps = [
        ":function_body",
        ":graph_constructor",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "function_utils",
    srcs = ["function_utils.cc"],
    hdrs = ["function_utils.h"],
    copts = tf_copts(),
    deps = [
        ":function_body",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "gradients",
    srcs = ["gradients.cc"],
    hdrs = ["gradients.h"],
    deps = [
        ":device",
        ":executor",
        ":graph_constructor",
        ":graph_optimizer",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
    ],
)

# This library also includes "eval_const_tensor", "graph_runner", and
# "shape_refiner", because there are circular dependencies between these
# modules.
cc_library(
    name = "graph_constructor",
    srcs = [
        "eval_const_tensor.cc",
        "graph_constructor.cc",
        "graph_runner.cc",
        "shape_refiner.cc",
        "//tensorflow/core/framework:versions.h",
    ],
    hdrs = [
        "eval_const_tensor.h",
        "graph_constructor.h",
        "graph_runner.h",
        "shape_refiner.h",
        ":core_cpu_lib_headers",
    ],
    copts = tf_copts(),
    visibility = default_package_visibility + [
        "//platforms/performance/autograppler:__subpackages__",
        "//platforms/performance/tf_sim:__subpackages__",
    ],
    deps = [
        ":device",
        ":device_factory",
        ":function_utils",
        ":memory_types",
        ":session_options",
        ":single_threaded_cpu_device",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:variant",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/tsl/framework:bfc_allocator",
        "//tensorflow/tsl/platform:mutex",
        "//tensorflow/tsl/platform:types",
    ] + if_static([
        ":executor",
        ":rendezvous_mgr",
    ]),
)

cc_library(
    name = "graph_def_builder_util",
    srcs = ["graph_def_builder_util.cc"],
    hdrs = ["graph_def_builder_util.h"],
    copts = tf_copts(),
    deps = [
        ":graph_constructor",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "graph_optimizer",
    srcs = ["graph_optimizer.cc"],
    hdrs = ["graph_optimizer.h"],
    copts = tf_copts(),
    deps = [
        ":constant_folding",
        ":function_utils",
        ":graph_constructor",
        ":inline_function_utils",
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
        "//tensorflow/core/profiler/lib:scoped_memory_debug_annotation",
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
    name = "inline_function_utils",
    srcs = ["inline_function_utils.cc"],
    hdrs = ["inline_function_utils.h"],
    copts = tf_copts(),
    deps = [
        ":device",
        ":function_body",
        ":function_utils",
        ":graph_constructor",
        ":lower_function_call_inline_policy",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/profiler/lib:traceme",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
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
    name = "isolate_placer_inspection_required_ops_pass",
    srcs = ["isolate_placer_inspection_required_ops_pass.cc"],
    hdrs = ["isolate_placer_inspection_required_ops_pass.h"],
    copts = tf_copts(),
    deps = [
        ":optimization_registry",
        ":placer_inspection_required_ops_utils",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
    ],
    alwayslink = 1,
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
        "//tensorflow/core/platform:cpu_feature_guard",
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
    name = "lower_case_op",
    srcs = ["lower_case_op.cc"],
    hdrs = ["lower_case_op.h"],
    deps = [
        ":inline_function_utils",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "lower_function_call_inline_policy",
    srcs = ["lower_function_call_inline_policy.cc"],
    hdrs = ["lower_function_call_inline_policy.h"],
    copts = tf_copts(),
    deps = [
        "//tensorflow/core:graph",
    ],
)

cc_library(
    name = "lower_function_call_op",
    srcs = ["lower_function_call_op.cc"],
    hdrs = ["lower_function_call_op.h"],
    copts = tf_copts(),
    deps = [
        ":function_def_utils",
        ":inline_function_utils",
        ":lower_function_call_inline_policy",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/algorithm:container",
    ],
)

cc_library(
    name = "lower_functional_ops",
    srcs = ["lower_functional_ops.cc"],
    hdrs = ["lower_functional_ops.h"],
    copts = tf_copts(),
    visibility = default_package_visibility + [
        "//platforms/performance/autograppler:__subpackages__",
        "//platforms/performance/tf_sim:__subpackages__",
    ],
    deps = [
        ":device_propagation",
        ":function_utils",
        ":inline_function_utils",
        ":lower_case_op",
        ":lower_function_call_op",
        ":lower_if_op",
        ":lower_while_op",
        ":optimization_registry",
        ":session_options",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/types:optional",
    ],
    alwayslink = 1,
)

cc_library(
    name = "int32_fulltype",
    srcs = ["int32_fulltype.cc"],
    hdrs = [
        "int32_fulltype.h",
        "optimization_registry.h",
    ],
    copts = tf_copts(),
    deps = [
        ":optimization_registry",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "lower_if_op",
    srcs = ["lower_if_op.cc"],
    hdrs = ["lower_if_op.h"],
    copts = tf_copts(),
    deps = [
        ":inline_function_utils",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "lower_while_op",
    srcs = ["lower_while_op.cc"],
    hdrs = ["lower_while_op.h"],
    copts = tf_copts(),
    deps = [
        ":inline_function_utils",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
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
    name = "layout_pass_util",
    srcs = ["layout_pass_util.cc"],
    hdrs = ["layout_pass_util.h"],
    copts = tf_copts(),
    deps = [
        "//tensorflow/core:graph",
        "//tensorflow/core:portable_gif_internal",
    ],
)

cc_library(
    name = "zen_layout_pass",
    srcs = ["zen_layout_pass.cc"],
    copts = tf_copts(),
    deps = [
        ":function",
        ":layout_pass_util",
        ":optimization_registry",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/graph:zen_graph_util",
        "@com_google_absl//absl/base",
    ],
    alwayslink = 1,
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
        "//tensorflow/core/util:onednn_env_vars",
    ],
)

cc_library(
    name = "mkl_layout_pass",
    srcs = ["mkl_layout_pass.cc"],
    hdrs = [
        "mkl_layout_pass.h",
        "//tensorflow/core/graph:mkl_graph_util_header",
    ],
    copts = tf_copts(),
    deps = [
        ":function",
        ":optimization_registry",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/base",
    ],
    alwayslink = 1,
)

cc_library(
    name = "mkl_tfconversion_pass",
    srcs = ["mkl_tfconversion_pass.cc"],
    hdrs = [
        "mkl_tfconversion_pass.h",
        "//tensorflow/core/graph:mkl_graph_util_header",
    ],
    copts = tf_copts(),
    deps = [
        ":function",
        ":optimization_registry",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
    alwayslink = 1,
)

cc_library(
    name = "node_file_writer",
    srcs = ["node_file_writer.cc"],
    hdrs = ["node_file_writer.h"],
    copts = tf_copts(),
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core/framework:attr_value_proto_cc",
        "//tensorflow/core/framework:node_def_proto_cc",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "optimization_registry",
    srcs = ["optimization_registry.cc"],
    hdrs = ["optimization_registry.h"],
    copts = tf_copts(),
    deps = [
        ":composite_device",
        ":device_set",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
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
        ":graph_constructor",
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
    name = "permuter",
    srcs = ["permuter.cc"],
    hdrs = ["permuter.h"],
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
        "//tensorflow/core/profiler/lib:traceme",
    ],
    alwayslink = 1,
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
        "//tensorflow/core/profiler/lib:traceme",
    ],
)

cc_library(
    name = "placer",
    srcs = ["placer.cc"],
    hdrs = ["placer.h"],
    copts = tf_copts(),
    deps = [
        ":colocation_graph",
        ":device",
        ":device_set",
        ":session_options",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "placer_inspection_required_ops_utils",
    srcs = ["placer_inspection_required_ops_utils.cc"],
    hdrs = ["placer_inspection_required_ops_utils.h"],
    copts = tf_copts(),
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
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
    linkopts = tf_openmp_lopts(),
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
        "//tensorflow/core/platform:hash",
        "//tensorflow/core/profiler/lib:traceme",
    ],
)

filegroup(
    name = "quantize_training_hdrs",
    srcs = [
        "quantize_training.h",
    ],
    visibility = [
        "//tensorflow/python:__pkg__",
    ],
)

cc_library(
    name = "quantize_training",
    srcs = ["quantize_training.cc"],
    hdrs = [":quantize_training_hdrs"],
    copts = tf_copts(),
    deps = [
        ":executor",
        ":graph_constructor",
        ":memory_types",
        ":session_options",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
    ],
    alwayslink = 1,
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
        "//tensorflow/core/profiler/lib:scoped_memory_debug_annotation",
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
        ":optimize_cross_host_control_deps",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "optimize_cross_host_control_deps",
    srcs = ["optimize_cross_host_control_deps.cc"],
    hdrs = ["optimize_cross_host_control_deps.h"],
    copts = tf_copts(),
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/strings",
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
    name = "local_session_selection",
    srcs = ["local_session_selection.cc"],
    hdrs = ["local_session_selection.h"],
    copts = tf_copts(),
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
    linkopts = tf_openmp_lopts(),
    deps = [
        ":device_factory",
        ":local_device",
        ":loose_headers",
        ":scoped_allocator",
        ":session_options",
        ":node_file_writer",
        "@com_google_absl//absl/base",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
    ] + if_mkl([":mkl_cpu_allocator"]) + if_mkl_ml([
        "//third_party/mkl:intel_binary_blob",
    ]),
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
    name = "core_cpu_impl",
    hdrs = [":core_cpu_lib_headers"],
    copts = tf_copts(),
    deps = [
        ":accumulate_n_optimizer",
        ":all_to_all",
        ":base_collective_executor",
        ":bfc_allocator",
        ":buf_rendezvous",
        ":build_graph_options",
        ":collective_executor_mgr",
        ":collective_param_resolver_local",
        ":collective_rma_local",
        ":collective_util",
        ":composite_device",
        ":copy_tensor",
        ":costmodel_manager",
        ":debugger_state_interface",
        ":device",
        ":device_factory",
        ":device_mgr",
        ":device_resolver_local",
        ":device_set",
        ":entry",
        ":function",
        ":graph_def_builder_util",
        ":graph_view",
        ":hierarchical_tree_broadcaster",
        ":input_colocation_exemption_registry",
        ":int32_fulltype",
        ":isolate_placer_inspection_required_ops_pass",
        ":local_device",
        ":lower_functional_ops",
        ":memory_types",
        ":mkl_cpu_allocator",
        ":mkl_layout_pass",
        ":mkl_tfconversion_pass",
        ":optimization_registry",
        ":optimized_function_graph_info",
        ":parallel_concat_optimizer",
        ":partitioning_utils",
        ":pending_counts",
        ":permuter",
        ":placer",
        ":pool_allocator",
        ":process_state",
        ":process_util",
        ":profile_handler",
        ":quantize_training",
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
    ] + if_zendnn([":zen_layout_pass"]),
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
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler/clusters:utils",
        "//tensorflow/core/grappler/clusters:virtual_cluster",
        "//tensorflow/core/grappler/optimizers:meta_optimizer",
        "//third_party/eigen3",
    ] + tf_additional_core_deps() + if_static([
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
        "allocator_retry.h",
    ],
    hdrs = ["bfc_allocator.h"],
    features = ["parse_headers"],
    visibility = ["//visibility:public"],
    deps = [
        ":shared_counter",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/profiler/lib:scoped_memory_debug_annotation",
        "//tensorflow/core/profiler/lib:traceme",
        "//tensorflow/tsl/framework:bfc_allocator",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "shared_counter",
    hdrs = ["shared_counter.h"],
    features = ["parse_headers"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/tsl/framework:shared_counter",
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
        ":local_session_selection",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/debug:debug_graph_utils",
        "//tensorflow/core/kernels:function_ops",
        "//tensorflow/core/nccl:collective_communicator",
        "//tensorflow/core/profiler/lib:connected_traceme",
        "//tensorflow/core/profiler/lib:device_profiler_session",
        "//tensorflow/core/profiler/lib:profiler_backends",
        "//tensorflow/core/profiler/lib:traceme_encode",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
    alwayslink = 1,
)

filegroup(
    name = "gpu_runtime_headers",
    srcs = [
        "device.h",
        "gpu_device_context.h",
    ],
)

cc_library(
    name = "cost_measurement",
    hdrs = ["cost_measurement.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "cost_measurement_registry",
    srcs = ["cost_measurement_registry.cc"],
    hdrs = ["cost_measurement_registry.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":cost_measurement",
        "//tensorflow/core:tflite_portable_logging",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "no_op_cost_measurement",
    srcs = ["no_op_cost_measurement.cc"],
    hdrs = ["no_op_cost_measurement.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":cost_constants",
        ":cost_measurement",
        ":cost_measurement_registry",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
)

cc_library(
    name = "request_cost",
    srcs = ["request_cost.cc"],
    hdrs = ["request_cost.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "request_cost_accessor",
    hdrs = ["request_cost_accessor.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":request_cost",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "request_cost_accessor_registry",
    srcs = ["request_cost_accessor_registry.cc"],
    hdrs = ["request_cost_accessor_registry.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":request_cost_accessor",
        "//tensorflow/core:tflite_portable_logging",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "null_request_cost_accessor",
    srcs = ["null_request_cost_accessor.cc"],
    hdrs = ["null_request_cost_accessor.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":request_cost_accessor_registry",
    ],
    alwayslink = 1,
)

cc_library(
    name = "cost_util",
    srcs = ["cost_util.cc"],
    hdrs = ["cost_util.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":cost_measurement",
        ":cost_measurement_registry",
        ":request_cost_accessor",
        ":request_cost_accessor_registry",
        "//tensorflow/core/platform:str_util",
    ],
)

cc_library(
    name = "cost_constants",
    hdrs = ["cost_constants.h"],
    visibility = ["//visibility:public"],
    deps = [],
)

cc_library(
    name = "device_propagation",
    srcs = ["device_propagation.cc"],
    hdrs = ["device_propagation.h"],
    deps = [
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core/platform:stringpiece",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

# For a more maintainable build this target should not exist and the headers
# should  be split into the existing cc_library targets, but this change was
# automatically  done so that we can remove long standing issues and complexity
# in the build system. It's up to the OWNERS of this package to get rid of it or
# not. The use of the textual_hdrs attribute is discouraged, use hdrs instead.
# Here it is used to avoid header parsing errors in packages where the feature
# parse_headers was enabled since loose headers were not being parsed. See
# go/loose-lsc-one-target-approach for more details.
cc_library(
    name = "loose_headers",
    tags = ["avoid_dep"],
    textual_hdrs = [
        "gpu_device_context.h",
        "mkl_cpu_allocator.h",
    ],
    visibility = [
        ":__pkg__",
        "//tensorflow/core/nccl:__pkg__",
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

tf_cuda_cc_test(
    name = "all_to_all_test",
    srcs = ["all_to_all_test.cc"],
    tags = ["no_cuda_on_cpu_tap"],
    deps = [
        ":collective_test_util",
        ":core",
        ":core_cpu",
        ":core_cpu_internal",
        ":process_util",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/framework:tensor_testutil",
        "//tensorflow/core/framework:types_proto_cc",
        "//tensorflow/core/platform:blocking_counter",
    ],
)

tf_cc_test(
    name = "composite_device_test",
    size = "small",
    srcs = [
        "composite_device_test.cc",
    ],
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
        "//tensorflow/core/nccl:collective_communicator",
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
        "@com_google_absl//absl/strings",
    ],
)

tf_cuda_cc_test(
    name = "ring_reducer_test",
    size = "small",
    srcs = [
        "ring_reducer_test.cc",
    ],
    tags = ["no_cuda_on_cpu_tap"],
    deps = [
        ":collective_test_util",
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
        "@com_google_absl//absl/memory",
    ],
)

tf_cuda_cc_test(
    name = "ring_gatherer_test",
    size = "small",
    srcs = [
        "ring_gatherer_test.cc",
    ],
    tags = ["no_cuda_on_cpu_tap"],
    deps = [
        ":collective_test_util",
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
        "@com_google_absl//absl/memory",
    ],
)

tf_cuda_cc_test(
    name = "hierarchical_tree_broadcaster_test",
    size = "small",
    srcs = [
        "hierarchical_tree_broadcaster_test.cc",
    ],
    tags = ["no_cuda_on_cpu_tap"],
    deps = [
        ":collective_test_util",
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
        "@com_google_absl//absl/memory",
    ],
)

tf_cuda_cc_test(
    name = "permuter_test",
    size = "small",
    srcs = [
        "permuter_test.cc",
    ],
    tags = ["no_cuda_on_cpu_tap"],
    deps = [
        ":collective_test_util",
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
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/types:span",
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
        "//tensorflow/core/util:onednn_env_vars",
    ],
)

tf_cuda_cc_test(
    name = "memory_types_test",
    size = "small",
    srcs = ["memory_types_test.cc"],
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
        "//tensorflow/core/kernels:cast_op",
        "//third_party/eigen3",
    ],
)

tf_cc_test(
    name = "constant_folding_test",
    size = "small",
    srcs = ["constant_folding_test.cc"],
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
        "//tensorflow/core/kernels:control_flow_ops",
        "//tensorflow/core/kernels:cwise_op",
        "//tensorflow/core/kernels:function_ops",
        "//tensorflow/core/kernels:resource_variable_ops",
    ],
)

tf_cc_test(
    name = "process_util_test",
    size = "small",
    srcs = ["process_util_test.cc"],
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
    name = "optimize_cross_host_control_deps_test",
    size = "small",
    srcs = ["optimize_cross_host_control_deps_test.cc"],
    deps = [
        ":optimize_cross_host_control_deps",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:cc_ops_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
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
    size = "medium",
    srcs = ["direct_session_test.cc"],
    args = [] + if_cuda(["--heap_check="]),  # The GPU tracer leaks memory
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
    ] + if_cuda(["//tensorflow/tsl/platform:cuda"]),
)

# This is identical to :common_runtime_direct_session_test with the addition of
# a dependency on alwayslink target //third_party/tensorflow/core/debug, which
# enables support for TensorFlow Debugger (tfdbg).
tf_cc_test(
    name = "direct_session_with_debug_test",
    srcs = ["direct_session_test.cc"],
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
    args = ["--heap_check="],  # The GPU tracer leaks memory
    tags = [
        "no_gpu",
        "nomac",  # TODO(b/250242601)
    ],
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
    ] + if_mkl(["//tensorflow/core:mkl_array_ops_op_lib"]),
)

tf_cc_test(
    name = "executor_test",
    size = "small",
    srcs = ["executor_test.cc"],
    deps = [
        ":core",
        ":core_cpu",
        ":core_cpu_internal",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:cc_ops_internal",
        "//tensorflow/cc:function_ops",
        "//tensorflow/cc:ops",
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
        "//tensorflow/core/kernels:function_ops",
        "//tensorflow/core/kernels:functional_ops",
        "//tensorflow/core/kernels:math",
        "//tensorflow/core/kernels:random_ops",
        "//tensorflow/core/kernels:relu_op",
        "//tensorflow/core/kernels:state",
    ],
)

tf_cc_test(
    name = "function_test",
    size = "small",
    srcs = ["function_test.cc"],
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
    name = "inline_function_utils_test",
    size = "small",
    srcs = ["inline_function_utils_test.cc"],
    deps = [
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
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
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
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

tf_cc_test_mkl(
    name = "mkl_related_tests",
    size = "small",
    srcs = [
        "mkl_layout_pass_test.cc",
        "mkl_tfconversion_pass_test.cc",
    ],
    linkstatic = 1,
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
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:scope",
        "//tensorflow/cc:sendrecv_ops",
        "//tensorflow/core/kernels:ops_util",
        "//third_party/eigen3",
    ] + if_mkl([
        "//tensorflow/core/kernels/mkl:mkl_aggregate_ops",
        "//tensorflow/core/kernels/mkl:mkl_batch_matmul_op",
        "//tensorflow/core/kernels/mkl:mkl_concat_op",
        "//tensorflow/core/kernels/mkl:mkl_conv_op",
        "//tensorflow/core/kernels/mkl:mkl_cwise_ops_common",
        "//tensorflow/core/kernels/mkl:mkl_dequantize_op",
        "//tensorflow/core/kernels/mkl:mkl_einsum_op",
        "//tensorflow/core/kernels/mkl:mkl_fused_batch_norm_op",
        "//tensorflow/core/kernels/mkl:mkl_identity_op",
        "//tensorflow/core/kernels/mkl:mkl_input_conversion_op",
        "//tensorflow/core/kernels/mkl:mkl_lrn_op",
        "//tensorflow/core/kernels/mkl:mkl_matmul_op",
        "//tensorflow/core/kernels/mkl:mkl_pooling_ops",
        "//tensorflow/core/kernels/mkl:mkl_qmatmul_op",
        "//tensorflow/core/kernels/mkl:mkl_quantize_op",
        "//tensorflow/core/kernels/mkl:mkl_relu_op",
        "//tensorflow/core/kernels/mkl:mkl_reshape_op",
        "//tensorflow/core/kernels/mkl:mkl_slice_op",
        "//tensorflow/core/kernels/mkl:mkl_softmax_op",
        "//tensorflow/core/kernels/mkl:mkl_tfconv_op",
        "//tensorflow/core/kernels/mkl:mkl_transpose_op",
        "//tensorflow/core/kernels/mkl:mkl_tmp_bf16_ops",
    ]),
)

# TODO(bmzhao): Refactor this target to use granular dependencies
# after stage 4 of the TF build refactor is complete:
# https://github.com/tensorflow/community/pull/179
tf_cc_test(
    name = "quantize_training_test",
    srcs = ["quantize_training_test.cc"],
    deps = [
        ":core_cpu",
        ":core_cpu_internal",
        ":direct_session_internal",
        "//tensorflow/core",
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
        "//tensorflow/core/util:protos_test_cc",
    ],
)

tf_cc_test(
    name = "graph_constructor_test",
    size = "small",
    srcs = ["graph_constructor_test.cc"],
    linkopts = select({
        "//tensorflow:macos": ["-headerpad_max_install_names"],
        "//conditions:default": [],
    }),
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

tf_cc_test(
    name = "cost_measurement_registry_test",
    srcs = ["cost_measurement_registry_test.cc"],
    deps = [
        ":cost_measurement_registry",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/time",
    ],
)

tf_cc_test(
    name = "no_op_cost_measurement_test",
    srcs = ["no_op_cost_measurement_test.cc"],
    deps = [
        ":no_op_cost_measurement",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "request_cost_test",
    srcs = ["request_cost_test.cc"],
    deps = [
        ":request_cost",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/time",
    ],
)

tf_cc_test(
    name = "request_cost_accessor_registry_test",
    srcs = ["request_cost_accessor_registry_test.cc"],
    deps = [
        ":request_cost_accessor",
        ":request_cost_accessor_registry",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/time",
    ],
)

tf_cc_test(
    name = "null_request_cost_accessor_test",
    srcs = ["null_request_cost_accessor_test.cc"],
    deps = [
        ":null_request_cost_accessor",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "cost_util_test",
    srcs = ["cost_util_test.cc"],
    deps = [
        ":cost_measurement_registry",
        ":cost_util",
        ":request_cost_accessor_registry",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "device_propagation_test",
    size = "small",
    srcs = ["device_propagation_test.cc"],
    deps = [
        ":device_propagation",
        "//tensorflow/cc:array_ops",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:cc_ops_internal",
        "//tensorflow/cc:scope",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_library(
    name = "optimized_function_graph_info",
    srcs = ["optimized_function_graph_info.cc"],
    hdrs = ["optimized_function_graph_info.h"],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":graph_constructor",
        "//tensorflow/core:framework",
        "//tensorflow/core/framework:optimized_function_graph_proto_cc",
    ],
)

tf_cc_test(
    name = "optimized_function_graph_info_test",
    srcs = ["optimized_function_graph_info_test.cc"],
    tags = if_oss([
        "no_oss",
    ]),  # b/169705709, no protobuf matchers in OSS.
    deps = [
        ":optimized_function_graph_info",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:testlib",
        "//tensorflow/core/framework:function_testlib",
        "//tensorflow/core/framework:optimized_function_graph_proto_cc",
        "//tensorflow/tsl/lib/core:status_test_util",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:status_matchers",
        "//tensorflow/tsl/platform:test",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "optimize_function_graph_utils",
    srcs = ["optimize_function_graph_utils.cc"],
    hdrs = ["optimize_function_graph_utils.h"],
    copts = tf_copts(),
    deps = [
        ":composite_device",
        ":device_set",
        ":function_body",
        ":function_optimization_registry",
        ":function_utils",
        ":optimization_registry",
        ":placer",
        ":replicate_per_replica_nodes",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/framework:graph_proto_cc",
    ],
)

tf_cc_test(
    name = "optimize_function_graph_utils_test",
    srcs = ["optimize_function_graph_utils_test.cc"],
    deps = [
        ":device",
        ":device_factory",
        ":device_set",
        ":optimize_function_graph_utils",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels:function_ops",
        "//tensorflow/tsl/lib/core:status_test_util",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:test",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_cc_test(
    name = "int32_fulltype_test",
    size = "small",
    srcs = [
        "int32_fulltype_test.cc",
    ],
    deps = [
        ":core",
        ":int32_fulltype",
        "//tensorflow/core:framework",
        "//tensorflow/core:ops",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/tsl/lib/core:status_test_util",
    ],
)

tf_cc_test(
    name = "arg_ret_placement_test",
    size = "small",
    srcs = [
        "arg_ret_placement_test.cc",
    ],
    deps = [
        ":arg_ret_placement",
        "//tensorflow/cc:scope",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_fuzz_test(
    name = "graph_constructor_fuzz",
    srcs = ["graph_constructor_fuzz.cc"],
    tags = ["no_oss"],
    deps = [
        ":graph_constructor",
        "//tensorflow/core/ops",
    ],
)

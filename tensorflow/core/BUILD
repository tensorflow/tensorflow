# Description:
# TensorFlow is a computational framework, primarily for use in machine
# learning applications.
#
# Public targets:
#
# ":protos_all_cc" - exports all core TensorFlow protos
#     ":protos_all_py" - py_proto_library version (Google-internal)
# ":lib" - exports the public non-test headers for:
#     //third_party/tensorflow/core/platform:: Platform-specific code and external dependencies
#     lib/: Low-level libraries that are not TensorFlow-specific
# ":test" - test equivalent of ":lib".
#     This is currently public, but may be made internal in the
#     future.  Try to avoid depending on it.
# ":framework" - exports the public non-test headers for:
#     util/: General low-level TensorFlow-specific libraries
#     framework/: Support for adding new ops & kernels
#     example/: Wrappers to simplify access to Example proto
# ":ops" - defines TensorFlow ops, but no implementations / kernels
#     ops/: Standard ops
#     user_ops/: User-supplied ops
#     This aggregates a number of smaller op libraries (":*_op_lib")
# ":core_cpu" - exports the public non-test headers for:
#     graph/: Support for graphs made up of ops
#     common_runtime/: Common code for execution of graphs
#     public/: Public APIs for running graphs
# ":core" - The code for ":core_cpu" plus a GPU runtime
# ":all_kernels" - The cpu-specific kernels, plus gpu kernels if
#     built with Cuda
# ":tensorflow_opensource" - The complete open-source package, including
#      ":all_kernels", ":core", and a Session implementation.
# ":tensorflow" - "tensorflow_opensource" plus some Google-internal libraries.
# ":testlib" - TensorFlow-specific test support, e.g. utilities for testing
#      kernels.
# ":direct_session" - An implementation of the Session interface that
#      directly runs Graphs via the internal TensorFlow executor.
#  "framework_lite" - Intended to be used by operator implementations
#      (kernels) that can also be run outside the tensorflow runtime. This
#      contains a small set of headers and utilities that can be used for core
#      kernels, without bringing in libraries that can bloat code size (e.g.,
#      logging is not included because using it will bring in a large amount of
#      ostream code).
#
# ":example_parser_configuration" -- A library for extracting the
#      tensorflow.Example proto configuration from a Graph.
#
# Public Android targets:
#
# filegroup ":android_proto_srcs" - Protos
# filegroup ":android_srcs" - Core sources
# cc_library ":android_tensorflow_lib" - Native library
# cc_library ":android_tensorflow_lib_lite" - Native library, without ops,
#   supporting SELECTIVE_REGISTRATION feature.
# portable_proto_library ":android_proto_lib" (Google-internal)
#
# Note that :framework and :lib have incomplete transitive dependencies (they
# declare but do not define some symbols) if framework_shared_object=True
# (meaning there is an explicit framework shared object). Missing symbols are
# included in //tensorflow:libtensorflow_framework.so. This split supports
# custom op registration; see comments on
# //tensorflow:libtensorflow_framework.so. It does mean that TensorFlow cc_test
# and cc_binary rules will not build. Using tf_cc_test and tf_cc_binary (from
# //tensorflow/tensorflow.bzl) will include the necessary symbols in binary
# build targets.

load(
    "//tensorflow:tensorflow.bzl",
    "cc_header_only_library",
    "if_android",
    "if_emscripten",
    "if_ios",
    "if_linux_x86_64",
    "if_mobile",
    "if_not_windows",
    "tf_android_core_proto_headers",
    "tf_android_core_proto_sources",
    "tf_cc_test",
    "tf_cc_test_mkl",
    "tf_cc_tests",
    "tf_copts",
    "tf_cuda_library",
    "tf_cuda_only_cc_test",
    "tf_features_nomodules_if_android",
    "tf_features_nomodules_if_emscripten",
    "tf_gen_op_libs",
    "tf_generate_proto_text_sources",
    "tf_genrule_cmd_append_to_srcs",
    "tf_openmp_copts",
    "tf_opts_nortti_if_android",
    "tf_opts_nortti_if_emscripten",
    "transitive_hdrs",
)
load("//tensorflow:tensorflow.bzl", "if_nccl")
load("//tensorflow:tensorflow.bzl", "tensorflow_opensource_extra_deps")

# load("//tensorflow:tensorflow.bzl", "tf_android_full_lite_protos")
load("//tensorflow:tensorflow.bzl", "tf_cc_test_gpu")
load("//tensorflow:tensorflow.bzl", "tf_cc_tests_gpu")
load("//tensorflow:tensorflow.bzl", "tf_cuda_cc_test")

# Placeholder: load("//tensorflow:tensorflow.bzl", "tf_portable_proto_lib")
load("//tensorflow:tensorflow.bzl", "tf_portable_proto_library")
load("//tensorflow:tensorflow.bzl", "tf_version_info_genrule")

# For platform specific build config
load(
    "//tensorflow/core/platform:default/build_config.bzl",
    "tf_additional_all_protos",
    "tf_additional_core_deps",
    "tf_additional_human_readable_json_deps",
    "tf_additional_lib_defines",
    "tf_additional_lib_deps",
    "tf_additional_libdevice_data",
    "tf_additional_libdevice_deps",
    "tf_additional_monitoring_hdrs",
    "tf_additional_numa_copts",
    "tf_additional_numa_deps",
    "tf_additional_numa_lib_defines",
    "tf_additional_test_deps",
    "tf_grpc_service_all",
    "tf_jspb_proto_library",
    "tf_kernel_tests_linkstatic",
    "tf_lib_proto_compiler_deps",
    "tf_lib_proto_parsing_deps",
    "tf_proto_library",
    "tf_proto_library_cc",
    "tf_protos_all",
    "tf_protos_all_impl",
    "tf_protos_grappler",
    "tf_protos_grappler_impl",
    "tf_pyclif_proto_library",
)
load(
    "//tensorflow/core/platform:default/build_config_root.bzl",
    "if_dynamic_kernels",
    "if_static",
    "tf_cuda_tests_tags",
    "tf_gpu_tests_tags",
)
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")
load("@local_config_tensorrt//:build_defs.bzl", "if_tensorrt")
load("@io_bazel_rules_closure//closure:defs.bzl", "closure_proto_library")
load(
    "//third_party/mkl:build_defs.bzl",
    "if_mkl",
    "mkl_deps",
)
# Placeholder for Google-internal load statements.

package(
    default_visibility = [
        ":dependency_whitelist",
        "//tensorflow:internal",
        "//tensorflow_models:__subpackages__",
    ],
    licenses = ["notice"],  # Apache 2.0
)

package_group(
    name = "dependency_whitelist",
    packages = [
        "//learning/freud/topic_models/tensorflow/...",
        "//perftools/accelerators/xprof/api/...",
        "//quality/webanswers/brain/tokenization/custom_tf_ops/kernels/...",
    ],
)

# Export the BUILD file so automated tooling can check licenses
exports_files([
    "BUILD",
    "framework/types.proto",
    "ops/ops.pbtxt",
])

package_group(name = "experimental_access")

# -----------------------------------------------------------------------------
# Public targets

# Protos which are needed for core tensorflow, including on mobile builds.
#
# Note that some protos are in neither additional_core_proto_srcs nor this
# filegroup; e.g.  ones with individual proto_library targets.
# LINT.IfChange
COMMON_PROTO_SRCS = [
    "example/example.proto",
    "example/feature.proto",
    "framework/allocation_description.proto",
    "framework/api_def.proto",
    "framework/attr_value.proto",
    "framework/cost_graph.proto",
    "framework/device_attributes.proto",
    "framework/function.proto",
    "framework/graph.proto",
    "framework/graph_transfer_info.proto",
    "framework/kernel_def.proto",
    "framework/log_memory.proto",
    "framework/node_def.proto",
    "framework/op_def.proto",
    "framework/reader_base.proto",
    "framework/remote_fused_graph_execute_info.proto",
    "framework/resource_handle.proto",
    "framework/step_stats.proto",
    "framework/summary.proto",
    "framework/tensor.proto",
    "framework/tensor_description.proto",
    "framework/tensor_shape.proto",
    "framework/tensor_slice.proto",
    "framework/types.proto",
    "framework/variable.proto",
    "framework/versions.proto",
    "protobuf/bfc_memory_map.proto",
    "protobuf/config.proto",
    "protobuf/cluster.proto",
    "protobuf/debug.proto",
    "protobuf/device_properties.proto",
    "protobuf/graph_debug_info.proto",
    "protobuf/queue_runner.proto",
    "protobuf/rewriter_config.proto",
    "protobuf/tensor_bundle.proto",
    "protobuf/saver.proto",
    "protobuf/verifier_config.proto",
    "protobuf/trace_events.proto",
    "util/event.proto",
    "util/memmapped_file_system.proto",
    "util/saved_tensor_slice.proto",
]

ERROR_CODES_PROTO_SRCS = [
    "protobuf/error_codes.proto",
    "//tensorflow/core/lib/core:error_codes.proto",
]
# LINT.ThenChange(//tensorflow/core/android_proto_config.asciipb)

CORE_PROTO_SRCS = COMMON_PROTO_SRCS + ERROR_CODES_PROTO_SRCS

# Protos which are not needed on mobile builds, but should be included in
# protos_all.
#
# Note that some protos are in neither core_proto_srcs nor this filegroup; e.g.
# ones with individual proto_library targets.
ADDITIONAL_CORE_PROTO_SRCS = [
    "example/example_parser_configuration.proto",
    "protobuf/trackable_object_graph.proto",
    "protobuf/control_flow.proto",
    "protobuf/data/experimental/snapshot.proto",
    # TODO(ebrevdo): Re-enable once CriticalSection is in core.
    # "protobuf/critical_section.proto",
    "protobuf/debug_event.proto",
    "protobuf/meta_graph.proto",
    "protobuf/named_tensor.proto",
    "protobuf/remote_tensor_handle.proto",
    "protobuf/saved_model.proto",
    "protobuf/saved_object_graph.proto",
    "protobuf/struct.proto",
    "protobuf/tensorflow_server.proto",
    "protobuf/transport_options.proto",
    "util/test_log.proto",
]

tf_proto_library(
    name = "protos_all",
    srcs = [],
    cc_api_version = 2,
    make_default_target_header_only = True,
    protodeps = [
        ":error_codes_proto_impl",
        ":protos_all_proto",
        "//tensorflow/core/lib/core:error_codes_proto",
    ],
    visibility = ["//visibility:public"],
)

tf_generate_proto_text_sources(
    name = "attr_value_proto_text",
    srcs = [
        "framework/attr_value.proto",
        "framework/resource_handle.proto",
        "framework/tensor.proto",
        "framework/tensor_shape.proto",
        "framework/types.proto",
    ],
    srcs_relative_dir = "tensorflow/core/",
    deps = [
        ":lib_internal",
        ":protos_all_proto_cc",
    ],
)

tf_jspb_proto_library(
    name = "protos_all_jspb_proto",
    visibility = ["//visibility:public"],
    deps = [":protos_all_cc"],
)

proto_library(
    name = "example_protos",
    srcs = [
        "example/example.proto",
        "example/feature.proto",
    ],
    visibility = ["//visibility:public"],
)

java_proto_library(
    name = "example_java_proto",
    visibility = ["//visibility:public"],
    deps = [":example_protos"],
)

closure_proto_library(
    name = "example_protos_closure",
    visibility = ["//visibility:public"],
    deps = [":example_protos"],
)

tf_proto_library(
    name = "protos_test",
    srcs = ["util/example_proto_fast_parsing_test.proto"],
    cc_api_version = 2,
    protodeps = tf_additional_all_protos(),
    visibility = ["//visibility:public"],
)

filegroup(
    name = "platform_base_hdrs",
    srcs = [
        "//tensorflow/core/platform:byte_order.h",
        "//tensorflow/core/platform:cord.h",
        "//tensorflow/core/platform:env_time.h",
        "//tensorflow/core/platform:logging.h",
        "//tensorflow/core/platform:macros.h",
        "//tensorflow/core/platform:platform_strings.h",
        "//tensorflow/core/platform:threadpool.h",
        "//tensorflow/core/platform:threadpool_interface.h",
        "//tensorflow/core/platform:threadpool_options.h",
        "//tensorflow/core/platform:tstring.h",
        "//tensorflow/core/platform:types.h",
    ],
    visibility = ["//visibility:private"],
)

cc_library(
    name = "platform_base",
    hdrs = [":platform_base_hdrs"],
    copts = tf_copts(),
    tags = ["avoid_dep"],
    visibility = [":__subpackages__"],
    deps = [
        "//tensorflow/core/platform",
        "//tensorflow/core/platform:byte_order",
        "//tensorflow/core/platform:env_time",
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/platform:macros",
        "//tensorflow/core/platform:types",
        "//tensorflow/core/platform/default/build_config:base",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "framework_bounds_check",
    hdrs = ["framework/bounds_check.h"],
    visibility = ["//tensorflow/core/kernels:friends"],
    deps = [
        ":platform_base",
        "//third_party/eigen3",
    ],
)

filegroup(
    name = "util_port_hdrs",
    srcs = [
        "util/port.h",
    ],
    visibility = [
        "//tensorflow/core:__pkg__",
        "//tensorflow/python:__pkg__",
    ],
)

cc_library(
    name = "util_port",
    srcs = ["util/port.cc"],
    hdrs = ["util/port.h"],
    copts = tf_copts(),
    visibility = [
        "//tensorflow/core:__pkg__",
        "//tensorflow/python:__pkg__",
    ],
)

filegroup(
    name = "platform_port_hdrs",
    srcs = [
        "//tensorflow/core/platform:cpu_info.h",
        "//tensorflow/core/platform:dynamic_annotations.h",
        "//tensorflow/core/platform:init_main.h",
        "//tensorflow/core/platform:mem.h",
        "//tensorflow/core/platform:mutex.h",
        "//tensorflow/core/platform:numa.h",
        "//tensorflow/core/platform:thread_annotations.h",
    ],
    visibility = ["//visibility:private"],
)

filegroup(
    name = "platform_protobuf_hdrs",
    srcs = [
        "//tensorflow/core/platform:protobuf.h",
    ],
    visibility = ["//visibility:private"],
)

cc_library(
    name = "grpc_services",
    srcs = [],
    hdrs = [
        "//tensorflow/core/platform:grpc_services.h",
    ],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = tf_grpc_service_all(),
)

cc_library(
    name = "human_readable_json",
    srcs = ["//tensorflow/core/platform:legacy_human_readable_json_src"],
    hdrs = ["//tensorflow/core/platform:human_readable_json.h"],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":lib",
        ":lib_internal",
    ] + tf_additional_human_readable_json_deps(),
)

cc_library(
    name = "logger",
    srcs = ["//tensorflow/core/platform:logger.cc"],
    hdrs = ["//tensorflow/core/platform:logger.h"],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":lib",
        ":lib_proto_parsing",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/synchronization",
    ],
)

filegroup(
    name = "platform_env_hdrs",
    srcs = [
        "//tensorflow/core/platform:env.h",
        "//tensorflow/core/platform:file_statistics.h",
        "//tensorflow/core/platform:file_system.h",
    ],
    visibility = ["//visibility:private"],
)

filegroup(
    name = "platform_file_system_hdrs",
    srcs = [
        "//tensorflow/core/platform:file_system_helper.h",
        "//tensorflow/core/platform:null_file_system.h",
    ],
    visibility = ["//visibility:private"],
)

filegroup(
    name = "platform_other_hdrs",
    srcs = [
        "//tensorflow/core/platform:abi.h",
        "//tensorflow/core/platform:context.h",
        "//tensorflow/core/platform:cpu_feature_guard.h",
        "//tensorflow/core/platform:error.h",
        "//tensorflow/core/platform:fingerprint.h",
        "//tensorflow/core/platform:monitoring.h",
        "//tensorflow/core/platform:net.h",
        "//tensorflow/core/platform:notification.h",
        "//tensorflow/core/platform:prefetch.h",
        "//tensorflow/core/platform:profile_utils/android_armv7a_cpu_utils_helper.h",
        "//tensorflow/core/platform:profile_utils/clock_cycle_profiler.h",
        "//tensorflow/core/platform:profile_utils/cpu_utils.h",
        "//tensorflow/core/platform:profile_utils/i_cpu_utils_helper.h",
        "//tensorflow/core/platform:stacktrace.h",
        "//tensorflow/core/platform:stacktrace_handler.h",
        "//tensorflow/core/platform:strong_hash.h",
        "//tensorflow/core/platform:subprocess.h",
    ] + tf_additional_monitoring_hdrs(),
    visibility = ["//visibility:private"],
)

tf_cc_test(
    name = "platform_unbounded_work_queue_test",
    srcs = ["//tensorflow/core/platform:unbounded_work_queue_test.cc"],
    deps = [
        ":framework",
        ":lib",
        ":lib_internal",
        ":lib_test_internal",
        ":test",
        ":test_main",
        "@com_google_absl//absl/memory",
    ],
)

# Minimal lib so that tools used for mobile compilation
# don't have to depend on lib/platformlib.
cc_library(
    name = "lib_proto_parsing",
    srcs = [
        "//tensorflow/core/platform:protobuf.cc",
    ],
    hdrs = [
        "//tensorflow/core/lib/bfloat16:bfloat16.h",
        "//tensorflow/core/lib/core:legacy_lib_proto_parsing_headers",
        "//tensorflow/core/lib/strings:legacy_lib_proto_parsing_headers",
        "//tensorflow/core/platform:init_main.h",
        "//tensorflow/core/platform:legacy_proto_hdrs",
        "//tensorflow/core/platform:logging.h",
        "//tensorflow/core/platform:macros.h",
        "//tensorflow/core/platform:platform.h",
        "//tensorflow/core/platform:protobuf.h",
        "//tensorflow/core/platform:stringpiece.h",
        "//tensorflow/core/platform:tstring.h",
        "//tensorflow/core/platform:types.h",
    ],
    copts = tf_copts(),
    deps = tf_lib_proto_parsing_deps() + [
        ":platform_base",
        "@com_google_absl//absl/strings",
        "@double_conversion//:double-conversion",
        "//tensorflow/core/lib/bfloat16",
        "//tensorflow/core/platform:cpu_info",
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/platform:macros",
        "//tensorflow/core/platform:platform",
        "//tensorflow/core/platform:stringpiece",
        "//tensorflow/core/platform:types",
    ],
)

cc_library(
    name = "lib_proto_compiler",
    hdrs = [
        "//tensorflow/core/platform:protobuf_compiler.h",
    ],
    copts = tf_copts(),
    deps = tf_lib_proto_compiler_deps() + [
        ":lib_proto_parsing",
    ],
)

# This build rule (along with :lib_internal, :framework, and
# :framework_internal) purposefully omits the definitions of many declared
# symbols, which are included in //tensorflow:libtensorflow_framework.so. Using
# tf_cc_test and tf_cc_binary will include the necessary symbols.
cc_library(
    name = "lib",
    hdrs = [
        "lib/histogram/histogram.h",
        "lib/monitoring/collected_metrics.h",
        "lib/monitoring/collection_registry.h",
        "lib/monitoring/counter.h",
        "lib/monitoring/gauge.h",
        "lib/monitoring/metric_def.h",
        "lib/monitoring/sampler.h",
        ":platform_base_hdrs",
        ":platform_env_hdrs",
        ":platform_file_system_hdrs",
        ":platform_other_hdrs",
        ":platform_port_hdrs",
        ":platform_protobuf_hdrs",
        "//tensorflow/core/lib/bfloat16:bfloat16.h",
        "//tensorflow/core/lib/core:legacy_lib_core_headers",
        "//tensorflow/core/lib/gtl:legacy_lib_gtl_headers",
        "//tensorflow/core/lib/hash:legacy_lib_hash_all_headers",
        "//tensorflow/core/lib/io:legacy_lib_io_headers",
        "//tensorflow/core/lib/math:math_util.h",
        "//tensorflow/core/lib/random:legacy_lib_random_headers",
        "//tensorflow/core/lib/strings:legacy_lib_string_headers",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":lib_internal",
        "//tensorflow/core/platform:str_util",
        "//tensorflow/core/platform:stringpiece",
        "//tensorflow/core/platform:stringprintf",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)

# APIs defined in lib_experimental are for experimental usage and may be
# subject to change. Its visibility is limited to selected packages.
cc_library(
    name = "lib_experimental",
    hdrs = [
        "//tensorflow/core/lib/core:legacy_lib_core_threadpool_options_header",
    ],
    visibility = [
        ":experimental_access",
        "//tensorflow/cc:__pkg__",
    ],
    deps = [
        ":lib",
    ],
)

cc_library(
    name = "feature_util",
    srcs = ["example/feature_util.cc"],
    hdrs = ["example/feature_util.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":core_stringpiece",
        ":lib_proto_parsing",
        ":protos_all_cc",
    ],
    alwayslink = 1,
)

# DEPRECATED: use platform:stringpiece instead.
cc_library(
    name = "core_stringpiece",
    hdrs = ["//tensorflow/core/lib/core:legacy_lib_core_stringpiece_header"],
    copts = tf_copts(),
    deps = [
        "//tensorflow/core/platform:stringpiece",
    ],
)

# Test support library needed for all tests
# This is currently public, but may be made internal in the
# future.  Try to avoid depending on it.
cc_library(
    name = "test",
    testonly = 1,
    srcs = [
        "util/reporter.cc",
        "//tensorflow/core/platform:legacy_test_srcs",
    ],
    hdrs = [
        "util/reporter.h",
        "//tensorflow/core/lib/core:legacy_lib_core_status_test_util_header",
        "//tensorflow/core/platform:test.h",
        "//tensorflow/core/platform:test_benchmark.h",
    ],
    copts = tf_copts(),
    linkopts = select({
        "//tensorflow:windows": [],
        "//conditions:default": ["-lm"],
    }),
    visibility = ["//visibility:public"],
    deps = [
        ":function_ops_op_lib",
        ":functional_ops_op_lib",
        ":lib",
        ":lib_internal",
        ":protos_all_cc",
        "//tensorflow/core/platform/default/build_config:gtest",
        "//tensorflow/core/kernels:required",
        "@com_google_googletest//:gtest",
    ] + tf_additional_test_deps(),
)

# This build rule (along with :framework_internal, :lib, and :lib_internal)
# purposefully omits the definitions of many declared symbols, which are
# included in //tensorflow:libtensorflow_framework.so. Using tf_cc_test and tf_cc_binary
# will include the necessary symbols.
tf_cuda_library(
    name = "framework",
    hdrs = [
        "example/feature_util.h",
        "framework/allocator.h",
        "framework/bounds_check.h",
        "framework/variant.h",
        "framework/variant_encode_decode.h",
        "framework/variant_op_registry.h",
        "framework/variant_tensor_data.h",
        "framework/shared_ptr_variant.h",
        "framework/allocator_registry.h",
        "framework/attr_value_util.h",
        "framework/bfloat16.h",
        "framework/cancellation.h",
        "framework/collective.h",
        "framework/common_shape_fns.h",
        "framework/control_flow.h",  # TODO(josh11b): Make internal?
        "framework/dataset.h",
        "framework/dataset_stateful_op_whitelist.h",
        "framework/device_base.h",
        "framework/function.h",
        "framework/function_handle_cache.h",
        "framework/graph_def_util.h",
        "framework/graph_to_functiondef.h",
        "framework/kernel_def_builder.h",
        "framework/kernel_def_util.h",
        "framework/log_memory.h",
        "framework/logging.h",
        "framework/lookup_interface.h",
        "framework/memory_types.h",
        "framework/node_def_builder.h",
        "framework/node_def_util.h",
        "framework/numeric_op.h",
        "framework/numeric_types.h",
        "framework/op.h",
        "framework/op_def_builder.h",
        "framework/op_def_util.h",
        "framework/op_kernel.h",
        "framework/ops_util.h",
        "framework/partial_tensor_shape.h",
        "framework/queue_interface.h",
        "framework/reader_interface.h",
        "framework/reader_op_kernel.h",
        "framework/register_types.h",
        "framework/register_types_traits.h",
        "framework/resource_mgr.h",
        "framework/resource_op_kernel.h",
        "framework/selective_registration.h",
        "framework/session_state.h",
        "framework/shape_inference.h",
        "framework/stats_aggregator.h",
        "framework/tensor.h",
        "framework/tensor_shape.h",
        "framework/tensor_slice.h",
        "framework/tensor_types.h",
        "framework/tensor_util.h",
        "framework/thread_factory.h",
        "framework/tracking_allocator.h",
        "framework/type_index.h",
        "framework/type_traits.h",
        "framework/typed_allocator.h",
        "framework/types.h",
        "public/version.h",
        "util/activation_mode.h",
        "util/batch_util.h",
        "util/bcast.h",
        "util/matmul_bcast.h",
        "util/debug_events_writer.h",
        "util/device_name_utils.h",
        "util/dump_graph.h",
        "util/events_writer.h",
        "util/example_proto_fast_parsing.h",
        "util/example_proto_helper.h",
        "util/gpu_kernel_helper.h",
        "util/guarded_philox_random.h",
        "util/mirror_pad_mode.h",
        "util/padding.h",
        "util/einsum_op_util.h",
        "util/port.h",
        "util/ptr_util.h",
        "util/reffed_status_callback.h",
        "util/saved_tensor_slice_util.h",
        "util/sparse/group_iterator.h",
        "util/sparse/sparse_tensor.h",
        "util/stat_summarizer.h",
        "util/stat_summarizer_options.h",
        "util/stream_executor_util.h",
        "util/strided_slice_op.h",
        "util/tensor_format.h",
        "util/tensor_ops_util.h",
        "util/tensor_slice_reader.h",
        "util/tensor_slice_reader_cache.h",
        "util/tensor_slice_writer.h",
        "util/use_cudnn.h",
        "util/matmul_autotune.h",
        "util/util.h",
        "util/work_sharder.h",
    ] + select({
        "//tensorflow:windows": [],
        "//conditions:default": [
            "util/memmapped_file_system.h",
            "util/memmapped_file_system_writer.h",
        ],
    }) + if_mkl([
        "util/mkl_util.h",
    ]),
    visibility = ["//visibility:public"],
    deps = [
        ":framework_internal",
        "@com_google_absl//absl/base",
    ],
)

# This is redundant with the "framework" target above. It's useful for
# applications that want to depend on a minimal subset of TensorFlow (e.g. XLA).
cc_library(
    name = "allocator",
    srcs = [
        "framework/allocator.cc",
        "framework/allocator_registry.h",
        "framework/numeric_types.h",
        "framework/tracking_allocator.cc",
        "framework/tracking_allocator.h",
        "framework/type_traits.h",
    ],
    hdrs = [
        "framework/allocator.h",
    ],
    features = ["parse_headers"],
    visibility = ["//visibility:public"],
    deps = [
        ":lib",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "//third_party/eigen3",
    ] + if_static(extra_deps = [":allocator_registry_impl"]),
    alwayslink = 1,
)

# This target will be included in libtensorflow_framework.so via the
# framework_internal_impl target.
# All other dependencies on this target need to go through if_static guard,
# as otherwise duplicate registration in the registry will cause crashes.
cc_library(
    name = "allocator_registry_impl",
    srcs = [
        "framework/allocator.h",
        "framework/allocator_registry.cc",
        "framework/allocator_registry.h",
        "framework/cpu_allocator_impl.cc",
        "framework/numeric_types.h",
        "framework/tracking_allocator.h",
        "framework/type_traits.h",
    ],
    deps = [
        ":lib",
        "//third_party/eigen3",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
    alwayslink = 1,
)

cc_library(
    name = "stats_calculator_portable",
    srcs = [
        "util/stat_summarizer_options.h",
        "util/stats_calculator.cc",
    ],
    hdrs = [
        "util/stats_calculator.h",
    ],
    copts = tf_copts(),
)

tf_cc_test(
    name = "stats_calculator_test",
    srcs = ["util/stats_calculator_test.cc"],
    deps = [
        ":stats_calculator_portable",
        ":test",
        ":test_main",
    ],
)

cc_library(
    name = "overflow",
    hdrs = ["util/overflow.h"],
    deps = [
        ":framework_lite",
        ":lib",
    ],
)

cc_library(
    name = "exec_on_stall",
    hdrs = ["util/exec_on_stall.h"],
    deps = [":framework_lite"],
)

cc_library(
    name = "ptr_util",
    hdrs = ["util/ptr_util.h"],
)

cc_library(
    name = "reader_base",
    srcs = ["framework/reader_base.cc"],
    hdrs = ["framework/reader_base.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":framework",
        ":lib",
        ":protos_all_cc",
    ],
)

cc_library(
    name = "op_gen_lib",
    srcs = ["framework/op_gen_lib.cc"],
    hdrs = ["framework/op_gen_lib.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":lib",
        ":lib_internal",
        ":protos_all_cc",
        "//tensorflow/core/util/proto:proto_utils",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "session_options",
    hdrs = ["public/session_options.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":lib",
        ":protos_all_cc",
    ],
)

cc_library(
    name = "framework_lite",
    hdrs = [
        "framework/numeric_types.h",
        "framework/tensor_types.h",
        "framework/type_traits.h",
        "//tensorflow/core/lib/bfloat16:bfloat16.h",
        "//tensorflow/core/platform:byte_order.h",
        "//tensorflow/core/platform:cpu_info.h",
        "//tensorflow/core/platform:default/integral_types.h",
        "//tensorflow/core/platform:default/logging.h",
        "//tensorflow/core/platform:dynamic_annotations.h",
        "//tensorflow/core/platform:macros.h",
        "//tensorflow/core/platform:mutex.h",
        "//tensorflow/core/platform:platform.h",
        "//tensorflow/core/platform:prefetch.h",
        "//tensorflow/core/platform:protobuf.h",
        "//tensorflow/core/platform:thread_annotations.h",
        "//tensorflow/core/platform:tstring.h",
        "//tensorflow/core/platform:types.h",
    ],
    visibility = ["//visibility:public"],
    deps =
        if_mobile([
            "@nsync//:nsync_cpp",
        ]) + [
            "//third_party/eigen3",
            "//tensorflow/core/lib/bfloat16",
            "//tensorflow/core/platform:dynamic_annotations",
            "//tensorflow/core/platform:thread_annotations",
            "//tensorflow/core/platform:mutex",
            "//tensorflow/core/platform/default/build_config:minimal",
            "//tensorflow/core/platform:types",
        ],
)

# Generates library per group of ops.
tf_gen_op_libs(
    is_external = False,
    op_lib_names = [
        "batch_ops",
        "bitwise_ops",
        "boosted_trees_ops",
        "tensor_forest_ops",
        "candidate_sampling_ops",
        "checkpoint_ops",
        "clustering_ops",
        "collective_ops",
        "control_flow_ops",
        "ctc_ops",
        "data_flow_ops",
        "dataset_ops",
        "decode_proto_ops",
        "encode_proto_ops",
        "experimental_dataset_ops",
        "function_ops",
        "functional_ops",
        "image_ops",
        "io_ops",
        "linalg_ops",
        "list_ops",
        "lookup_ops",
        "logging_ops",
        "manip_ops",
        "math_ops",
        "mkl_nn_ops",
        "nccl_ops",
        "nn_ops",
        "no_op",
        "parsing_ops",
        "random_grad",
        "random_ops",
        "stateful_random_ops",
        "remote_fused_graph_ops",
        "rnn_ops",
        "rpc_ops",
        "scoped_allocator_ops",
        "sdca_ops",
        "set_ops",
        "script_ops",
        "sendrecv_ops",
        "sparse_csr_matrix_ops",
        "sparse_ops",
        "spectral_ops",
        "state_ops",
        "stateless_random_ops",
        "summary_ops",
        "training_ops",
    ],
    deps = [
        ":lib",
        ":protos_all_cc",
    ],
)

tf_gen_op_libs(
    op_lib_names = [
        "string_ops",
    ],
    deps = [
        ":lib_internal",
        ":lib_proto_parsing",
        "@com_google_absl//absl/strings",
    ],
)

tf_gen_op_libs(
    op_lib_names = [
        "array_ops",
    ],
    deps = [
        ":lib",
        ":protos_all_cc",
    ],
)

tf_gen_op_libs(
    op_lib_names = [
        "mkl_array_ops",
    ],
    deps = [":protos_all_cc"],
)

tf_gen_op_libs(
    op_lib_names = [
        "audio_ops",
    ],
    deps = [":lib"],
)

tf_gen_op_libs(
    op_lib_names = ["debug_ops"],
    deps = [":lib"],
)

tf_gen_op_libs(
    is_external = False,
    op_lib_names = [
        "resource_variable_ops",
    ],
    deps = [":lib"],
)

tf_gen_op_libs(
    op_lib_names = [
        "tpu_configuration_ops",
        "tpu_cross_replica_ops",
        "tpu_embedding_ops",
        "tpu_functional_ops",
        "tpu_heartbeat_ops",
        "tpu_host_compute_ops",
        "tpu_infeed_ops",
        "tpu_outfeed_ops",
        "tpu_ordinal_selector_ops",
        "tpu_replication_ops",
    ],
    deps = [
        ":lib",
        ":lib_proto_parsing",
        ":protos_all_cc",
        "//tensorflow/core/protobuf/tpu:tpu_embedding_configuration_proto_cc",
        "//tensorflow/core/tpu:tpu_embedding_optimization_parameters_utils",
        "//tensorflow/core/tpu:tpu_embedding_output_layout_utils",
    ],
)

# And one for all user ops
cc_library(
    name = "user_ops_op_lib",
    srcs = glob(["user_ops/**/*.cc"]),
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [":framework"],
    alwayslink = 1,
)

cc_library(
    name = "word2vec_ops",
    srcs = ["ops/word2vec_ops.cc"],
    linkstatic = 1,
    visibility = ["//tensorflow:internal"],
    deps = [":framework"],
    alwayslink = 1,
)

cc_library(
    name = "cudnn_rnn_ops",
    srcs = [
        "ops/cudnn_rnn_ops.cc",
    ],
    linkstatic = 1,
    visibility = ["//tensorflow:internal"],
    deps = [
        ":framework",
        ":lib",
        ":lib_internal",
        ":stream_executor",
        "//tensorflow/core/kernels:bounds_check_lib",
    ],
    alwayslink = 1,
)

tf_gen_op_libs(
    op_lib_names = [
        "cudnn_rnn_ops",
    ],
    deps = [
        ":lib",
    ],
)

cc_library(
    name = "ragged_ops",
    deps = [
        ":ragged_array_ops_op_lib",
        ":ragged_conversion_ops_op_lib",
        ":ragged_math_ops_op_lib",
    ],
)

tf_gen_op_libs(
    op_lib_names = [
        "ragged_array_ops",
        "ragged_conversion_ops",
        "ragged_math_ops",
    ],
    deps = [":ragged_to_dense_util"],
)

cc_library(
    name = "ragged_to_dense_util",
    srcs = [
        "ops/ragged_to_dense_util.cc",
    ],
    hdrs = [
        "ops/ragged_to_dense_util.h",
    ],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
    ],
)

tf_cc_test(
    name = "ragged_to_dense_util_test",
    srcs = [
        "ops/ragged_to_dense_util_test.cc",
    ],
    deps = [
        ":ragged_to_dense_util",
        ":test",
        ":testlib",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "ops",
    visibility = ["//visibility:public"],
    deps = [
        ":array_ops_op_lib",
        ":audio_ops_op_lib",
        ":batch_ops_op_lib",
        ":bitwise_ops_op_lib",
        ":boosted_trees_ops_op_lib",
        ":tensor_forest_ops_op_lib",
        ":candidate_sampling_ops_op_lib",
        ":checkpoint_ops_op_lib",
        ":clustering_ops_op_lib",
        ":collective_ops_op_lib",
        ":control_flow_ops_op_lib",
        ":ctc_ops_op_lib",
        ":cudnn_rnn_ops_op_lib",
        ":data_flow_ops_op_lib",
        ":dataset_ops_op_lib",
        ":debug_ops_op_lib",
        ":decode_proto_ops_op_lib",
        ":encode_proto_ops_op_lib",
        ":experimental_dataset_ops_op_lib",
        ":function_ops_op_lib",
        ":functional_ops_op_lib",
        ":image_ops_op_lib",
        ":io_ops_op_lib",
        ":linalg_ops_op_lib",
        ":list_ops_op_lib",
        ":logging_ops_op_lib",
        ":lookup_ops_op_lib",
        ":manip_ops_op_lib",
        ":math_ops_op_lib",
        ":nccl_ops_op_lib",
        ":nn_ops_op_lib",
        ":no_op_op_lib",
        ":parsing_ops_op_lib",
        ":ragged_ops",
        ":random_ops_op_lib",
        ":rnn_ops_op_lib",
        ":stateful_random_ops_op_lib",
        ":remote_fused_graph_ops_op_lib",
        ":resource_variable_ops_op_lib",
        ":rpc_ops_op_lib",
        ":scoped_allocator_ops_op_lib",
        ":script_ops_op_lib",
        ":sdca_ops_op_lib",
        ":sendrecv_ops_op_lib",
        ":set_ops_op_lib",
        ":sparse_csr_matrix_ops_op_lib",
        ":sparse_ops_op_lib",
        ":summary_ops_op_lib",
        ":spectral_ops_op_lib",
        ":state_ops_op_lib",
        ":stateless_random_ops_op_lib",
        ":string_ops_op_lib",
        ":tpu_configuration_ops_op_lib",
        ":tpu_cross_replica_ops_op_lib",
        ":tpu_embedding_ops_op_lib",
        ":tpu_functional_ops_op_lib",
        ":tpu_heartbeat_ops_op_lib",
        ":tpu_host_compute_ops_op_lib",
        ":tpu_infeed_ops_op_lib",
        ":tpu_outfeed_ops_op_lib",
        ":tpu_ordinal_selector_ops_op_lib",
        ":tpu_replication_ops_op_lib",
        ":training_ops_op_lib",
        ":user_ops_op_lib",
        ":word2vec_ops",
        "//tensorflow/c/kernels:bitcast_op_lib",
        "//tensorflow/compiler/mlir/tensorflow:mlir_passthrough_op",
    ] + if_mkl([
        ":mkl_array_ops_op_lib",
        ":mkl_nn_ops_op_lib",
    ]) + if_tensorrt([
        "//tensorflow/compiler/tf2tensorrt:trt_engine_resource_ops_op_lib",
        "//tensorflow/compiler/tf2tensorrt:trt_op_libs",
    ]),
    alwayslink = 1,
)

cc_library(
    name = "array_grad",
    srcs = ["ops/array_grad.cc"],
    linkstatic = 1,  # Needed since alwayslink is broken in bazel b/27630669
    visibility = ["//visibility:public"],
    deps = [
        ":array_ops_op_lib",
        ":framework",
        ":lib",
        "//tensorflow/c/kernels:bitcast_op_lib",
    ],
    alwayslink = 1,
)

cc_library(
    name = "functional_grad",
    srcs = ["ops/functional_grad.cc"],
    linkstatic = 1,  # Needed since alwayslink is broken in bazel b/27630669
    visibility = ["//visibility:public"],
    deps = [
        ":framework",
        ":functional_ops_op_lib",
        ":lib",
    ],
    alwayslink = 1,
)

cc_library(
    name = "math_grad",
    srcs = [
        "ops/math_grad.cc",
        "ops/random_grad.cc",
        "ops/stateless_random_grad.cc",
    ],
    linkstatic = 1,  # Needed since alwayslink is broken in bazel b/27630669
    visibility = ["//visibility:public"],
    deps = [
        ":framework",
        ":lib",
        ":math_ops_op_lib",
        ":protos_all_cc",
    ],
    alwayslink = 1,
)

cc_library(
    name = "nn_grad",
    srcs = ["ops/nn_grad.cc"],
    linkstatic = 1,  # Needed since alwayslink is broken in bazel b/27630669
    visibility = ["//visibility:public"],
    deps = [
        ":framework",
        ":lib",
        ":nn_ops_op_lib",
    ] + if_mkl([
        ":mkl_nn_ops_op_lib",
    ]),
    alwayslink = 1,
)

tf_cuda_library(
    name = "core_cpu",
    hdrs = [
        "common_runtime/device.h",
        "common_runtime/device_factory.h",
        "common_runtime/function.h",
        "common_runtime/optimization_registry.h",
        "common_runtime/shape_refiner.h",
        "graph/algorithm.h",
        "graph/default_device.h",
        "graph/gradients.h",
        "graph/graph.h",
        "graph/graph_constructor.h",
        "graph/graph_def_builder.h",
        "graph/graph_def_builder_util.h",
        "graph/node_builder.h",
        "graph/validate.h",
        "graph/while_context.h",
        "public/session.h",
        "public/session_options.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":core_cpu_internal",
    ],
)

cc_library(
    name = "core",
    visibility = ["//visibility:public"],
    deps = [
        ":core_cpu",
        ":gpu_runtime",
        ":sycl_runtime",
    ],
)

# This includes implementations of all kernels built into TensorFlow.
cc_library(
    name = "all_kernels_impl",
    visibility = [":__subpackages__"],
    deps = [
        "//tensorflow/c/kernels:bitcast_op",
        "//tensorflow/core/kernels:array",
        "//tensorflow/core/kernels:audio",
        "//tensorflow/core/kernels:batch_kernels",
        "//tensorflow/core/kernels:bincount_op",
        "//tensorflow/core/kernels:boosted_trees_ops",
        "//tensorflow/core/kernels:tensor_forest_ops",
        "//tensorflow/core/kernels:candidate_sampler_ops",
        "//tensorflow/core/kernels:checkpoint_ops",
        "//tensorflow/core/kernels:clustering_ops",
        "//tensorflow/core/kernels:collective_ops",
        "//tensorflow/core/kernels:control_flow_ops",
        "//tensorflow/core/kernels:ctc_ops",
        "//tensorflow/core/kernels:cudnn_rnn_kernels",
        "//tensorflow/core/kernels:data_flow",
        "//tensorflow/core/kernels:decode_proto_op",
        "//tensorflow/core/kernels:encode_proto_op",
        "//tensorflow/core/kernels:fake_quant_ops",
        "//tensorflow/core/kernels:function_ops",
        "//tensorflow/core/kernels:functional_ops",
        "//tensorflow/core/kernels:grappler",
        "//tensorflow/core/kernels:histogram_op",
        "//tensorflow/core/kernels:image",
        "//tensorflow/core/kernels:io",
        "//tensorflow/core/kernels:linalg",
        "//tensorflow/core/kernels:lookup",
        "//tensorflow/core/kernels:logging",
        "//tensorflow/core/kernels:manip",
        "//tensorflow/core/kernels:math",
        "//tensorflow/core/kernels:multinomial_op",
        "//tensorflow/core/kernels:mutex_ops",
        "//tensorflow/core/kernels:nn",
        "//tensorflow/core/kernels:parameterized_truncated_normal_op",
        "//tensorflow/core/kernels:parsing",
        "//tensorflow/core/kernels:partitioned_function_ops",
        "//tensorflow/core/kernels:pooling_ops",
        "//tensorflow/core/kernels:ragged_ops",
        "//tensorflow/core/kernels:random_ops",
        "//tensorflow/core/kernels:stateful_random_ops",
        "//tensorflow/core/kernels:random_binomial_op",
        "//tensorflow/core/kernels:random_poisson_op",
        "//tensorflow/core/kernels:remote_fused_graph_ops",
        "//tensorflow/core/kernels:required",
        "//tensorflow/core/kernels:resource_variable_ops",
        "//tensorflow/core/kernels:rnn_ops",
        "//tensorflow/core/kernels:rpc_op",
        "//tensorflow/core/kernels:scoped_allocator_ops",
        "//tensorflow/core/kernels:sdca_ops",
        "//tensorflow/core/kernels:searchsorted_op",
        "//tensorflow/core/kernels:set_kernels",
        "//tensorflow/core/kernels:sparse",
        "//tensorflow/core/kernels:state",
        "//tensorflow/core/kernels:stateless_random_ops",
        "//tensorflow/core/kernels:string",
        "//tensorflow/core/kernels:summary_kernels",
        "//tensorflow/core/kernels:training_ops",
        "//tensorflow/core/kernels:word2vec_kernels",
        "//tensorflow/core/kernels/sparse:kernels",
    ] + if_not_windows([
        "//tensorflow/core/kernels:fact_op",
        "//tensorflow/core/kernels:array_not_windows",
        "//tensorflow/core/kernels:math_not_windows",
        "//tensorflow/core/kernels:quantized_ops",
        "//tensorflow/core/kernels/neon:neon_depthwise_conv_op",
    ]) + if_mkl([
        "//tensorflow/core/kernels:mkl_aggregate_ops",
        "//tensorflow/core/kernels:mkl_concat_op",
        "//tensorflow/core/kernels:mkl_dequantize_op",
        "//tensorflow/core/kernels:mkl_conv_op",
        "//tensorflow/core/kernels:mkl_cwise_ops_common",
        "//tensorflow/core/kernels:mkl_fused_batch_norm_op",
        "//tensorflow/core/kernels:mkl_identity_op",
        "//tensorflow/core/kernels:mkl_input_conversion_op",
        "//tensorflow/core/kernels:mkl_lrn_op",
        "//tensorflow/core/kernels:mkl_pooling_ops",
        "//tensorflow/core/kernels:mkl_qmatmul_op",
        "//tensorflow/core/kernels:mkl_requantize_ops",
        "//tensorflow/core/kernels:mkl_quantize_op",
        "//tensorflow/core/kernels:mkl_relu_op",
        "//tensorflow/core/kernels:mkl_reshape_op",
        "//tensorflow/core/kernels:mkl_slice_op",
        "//tensorflow/core/kernels:mkl_softmax_op",
        "//tensorflow/core/kernels:mkl_transpose_op",
        "//tensorflow/core/kernels:mkl_batch_matmul_op",
        "//tensorflow/core/kernels:mkl_matmul_op",
        "//tensorflow/core/kernels:mkl_tfconv_op",
    ]) + if_cuda([
        "//tensorflow/core/grappler/optimizers:gpu_swapping_kernels",
        "//tensorflow/core/grappler/optimizers:gpu_swapping_ops",
    ]) + if_nccl([
        "//tensorflow/core/kernels:nccl_kernels",
    ]) + if_tensorrt([
        "//tensorflow/compiler/tf2tensorrt:trt_engine_resource_op_kernels",
        "//tensorflow/compiler/tf2tensorrt:trt_op_kernels",
    ]),
)

cc_library(
    name = "all_kernels",
    visibility = ["//visibility:public"],
    deps = if_dynamic_kernels(
        [],
        otherwise = [":all_kernels_impl"],
    ) + [
        # TODO(gunan): Work on the API between these and rest of TF and make
        # these also dynamically loading.
        "//tensorflow/core/kernels:dataset_ops",  # Depends on grappler
        "//tensorflow/core/kernels:list_kernels",  # Depends on variant_op_registry.h
    ],
)

tf_cuda_library(
    name = "tensorflow_opensource",
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":all_kernels",
        ":core",
        ":direct_session",
        ":example_parser_configuration",
        ":gpu_runtime",
        ":lib",
        ":ops",
    ] + tensorflow_opensource_extra_deps(),
)

cc_library(
    name = "tensorflow",
    visibility = ["//visibility:public"],
    deps = [
        ":tensorflow_opensource",
        "//tensorflow/core/platform/default/build_config:tensorflow_platform_specific",
    ],
)

# Test support library needed for higher-level (TensorFlow-specific) tests
cc_library(
    name = "testlib",
    testonly = 1,
    srcs = [
        "common_runtime/function_testlib.cc",
        "common_runtime/kernel_benchmark_testlib.cc",
        "framework/fake_input.cc",
        "framework/function_testlib.cc",
        "graph/testlib.cc",
    ],
    hdrs = [
        "common_runtime/function_testlib.h",
        "common_runtime/kernel_benchmark_testlib.h",
        "common_runtime/test_collective_executor_mgr.h",
        "framework/fake_input.h",
        "framework/function_testlib.h",
        "framework/shape_inference_testutil.h",
        "framework/tensor_testutil.h",
        "graph/benchmark_testlib.h",
        "graph/testlib.h",
        # TODO(josh11b): Drop this once users are depending on
        # kernels:ops_testutil instead.
        "//tensorflow/core/kernels:ops_testutil.h",
    ],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":core_cpu",
        ":core_cpu_internal",
        ":core_cpu_lib",
        ":framework",
        ":framework_internal",
        ":lib",
        ":lib_internal",
        ":ops",
        ":protos_all_cc",
        ":shape_inference_testutil",
        ":tensor_testutil",
        ":test",
        ":testlib_ops",
        "//tensorflow/cc:scope",
        "//tensorflow/core/kernels:ops_testutil",
        "//tensorflow/core/kernels:ops_util",
    ] + if_dynamic_kernels(
        [],
        otherwise = [
            "//tensorflow/core/kernels:aggregate_ops",
            "//tensorflow/core/kernels:bcast_ops",
            "//tensorflow/core/kernels:cast_op",
            "//tensorflow/core/kernels:constant_op",
            "//tensorflow/core/kernels:identity_op",
            "//tensorflow/core/kernels:random_ops",
            "//tensorflow/core/kernels:reduction_ops",
            "//tensorflow/core/kernels:reshape_op",
        ],
    ),
)

cc_library(
    name = "testlib_ops",
    testonly = 1,
    srcs = ["common_runtime/testlib_ops.cc"],
    linkstatic = 1,  # Seems to be needed since alwayslink is broken in bazel
    deps = [
        ":framework",
        ":lib",
    ],
    alwayslink = 1,
)

# This is a link-only library to provide a DirectSession
# implementation of the Session interface.
tf_cuda_library(
    name = "direct_session",
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":direct_session_internal",
    ],
    alwayslink = 1,
)

# -----------------------------------------------------------------------------
# MKL targets
cc_library(
    name = "mkl_graph_util",
    hdrs = ["graph/mkl_graph_util.h"],
)

# -----------------------------------------------------------------------------
# Public Android targets

# List of protos we want on android
filegroup(
    name = "android_proto_srcs",
    srcs = tf_android_core_proto_sources(CORE_PROTO_SRCS),
    visibility = ["//visibility:public"],
)

# Core sources for Android builds.
filegroup(
    name = "mobile_srcs_no_runtime",
    srcs = [
        ":attr_value_proto_text_srcs",
        "//tensorflow/core/platform/default/build_config:android_srcs",
        "//tensorflow/core/util/ctc:android_srcs",
        "//tensorflow/core/platform:legacy_srcs_no_runtime",
        "//tensorflow/core/profiler:mobile_srcs",
        "//tensorflow/core/lib/bfloat16:bfloat16.h",
        "//tensorflow/core/lib/bfloat16:bfloat16.cc",
        "//tensorflow/core/lib/core:legacy_lib_core_all_headers",
        "//tensorflow/core/lib/core:legacy_lib_core_all_srcs",
        "//tensorflow/core/lib/gtl:legacy_lib_gtl_all_headers",
        "//tensorflow/core/lib/hash:legacy_lib_hash_all_headers",
        "//tensorflow/core/lib/hash:legacy_lib_hash_all_srcs",
        "//tensorflow/core/lib/io:legacy_lib_io_all_headers",
        "//tensorflow/core/lib/io:legacy_lib_io_all_srcs",
        "//tensorflow/core/lib/random:legacy_lib_random_all_headers",
        "//tensorflow/core/lib/random:legacy_lib_random_all_srcs",
        "//tensorflow/core/lib/strings:legacy_lib_strings_all_headers",
        "//tensorflow/core/lib/strings:legacy_lib_strings_all_srcs",
        "//tensorflow/core/lib/math:math_util.h",
    ] + glob(
        [
            "client/**/*.cc",
            "framework/**/*.h",
            "framework/**/*.cc",
            "lib/**/*.h",
            "lib/**/*.cc",
            "public/**/*.h",
            "util/**/*.h",
            "util/**/*.cc",
        ],
        exclude = [
            "**/*test.*",
            "**/*testutil*",
            "**/*testlib*",
            "**/*main.cc",
            "debug/**/*",
            "framework/op_gen_*",
            "framework/node_def_util.*",
            "framework/op_kernel.*",
            "framework/dataset.*",
            "lib/jpeg/**/*",
            "lib/png/**/*",
            "lib/gif/**/*",
            "util/debug_events_writer.*",
            "util/events_writer.*",
            "util/stats_calculator.*",
            "util/reporter.*",
            "user_ops/**/*.cu.cc",
            "util/ctc/*.h",
            "util/ctc/*.cc",
            "util/tensor_bundle/*.h",
            "util/tensor_bundle/*.cc",
            "common_runtime/gpu/**/*",
            "common_runtime/eager/*",
            "common_runtime/gpu_device_factory.*",
        ],
    ),
    visibility = ["//visibility:private"],
)

filegroup(
    name = "mobile_srcs_only_runtime",
    srcs = [
        "//tensorflow/core/common_runtime/eager:srcs",
        "//tensorflow/core/kernels:android_srcs",
        "//tensorflow/core/util/ctc:android_srcs",
        "//tensorflow/core/util/tensor_bundle:android_srcs",
        "//tensorflow/c:srcs",
        "//tensorflow/c/eager:srcs",
    ] + glob(
        [
            "common_runtime/**/*.h",
            "common_runtime/**/*.cc",
            "graph/**/*.h",
            "graph/**/*.cc",
            "framework/node_def_util.*",
            "framework/op_kernel.*",
            "framework/dataset.*",
        ],
        exclude = [
            "**/*test.*",
            "**/*testutil*",
            "**/*testlib*",
            "**/*main.cc",
            "common_runtime/gpu/**/*",
            "common_runtime/gpu_device_factory.*",
            "graph/dot.*",
        ],
    ),
    visibility = ["//visibility:public"],
)

filegroup(
    name = "mobile_srcs",
    srcs = [
        ":mobile_srcs_no_runtime",
        ":mobile_srcs_only_runtime",
    ],
    visibility = ["//visibility:public"],
)

# Native library support for Android applications.  Does not contain
# operators, use :android_tensorflow_lib if you want full operator
# support.
#
# If you just need TensorFlow types, e.g. Tensors, use
# :android_tensorflow_lib_lite_no_runtime.
#
# Compiles to a trivial library on non-Android to prevent irrelevant
# build errors. If not building this as part of an android_binary,
# a command such as the following must be used:
# bazel build -c opt tensorflow/core:android_tensorflow_lib \
# --crosstool_top=//external:android/crosstool \
# --cpu=armeabi-v7a \
# --host_crosstool_top=@bazel_tools//tools/cpp:toolchain
cc_library(
    name = "android_tensorflow_lib_lite",
    srcs = if_android([":android_srcs"]),
    copts = tf_copts(android_optimization_level_override = None) + [
        "-DSUPPORT_SELECTIVE_REGISTRATION",
    ],
    linkopts = ["-lz"],
    tags = [
        "manual",
        "notap",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":mobile_additional_lib_deps",
        ":protos_all_cc_impl",
        ":stats_calculator_portable",
        "//third_party/eigen3",
        "@com_google_protobuf//:protobuf",
        "@double_conversion//:double-conversion",
        "@farmhash_archive//:farmhash",
        "@nsync//:nsync_cpp",
    ],
    alwayslink = 1,
)

cc_library(
    name = "android_tensorflow_lib_lite_nortti",
    srcs = if_android([":android_srcs"]),
    copts = tf_copts(android_optimization_level_override = None) + [
        "-DSUPPORT_SELECTIVE_REGISTRATION",
    ] + tf_opts_nortti_if_android(),
    linkopts = ["-lz"],
    tags = [
        "manual",
        "notap",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":mobile_additional_lib_deps",
        ":protos_all_cc_impl",
        ":stats_calculator_portable",
        "//third_party/eigen3",
        "@com_google_protobuf//:protobuf",
        "@double_conversion//:double-conversion",
        "@farmhash_archive//:farmhash",
        "@nsync//:nsync_cpp",
    ],
    alwayslink = 1,
)

cc_library(
    name = "mobile_additional_lib_deps",
    deps = tf_additional_lib_deps() + [
        ":platform_base",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "emscripten_tensorflow_lib_lite_nortti_lite_protos_no_runtime",
    srcs = if_emscripten([":mobile_srcs_no_runtime"]),
    copts = ["-DSUPPORT_SELECTIVE_REGISTRATION"] + tf_opts_nortti_if_emscripten(),
    defines = ["TENSORFLOW_LITE_PROTOS"],
    tags = [
        "manual",
        "notap",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":emscripten_proto_lib_no_rtti_lite_runtime",
        ":mobile_additional_lib_deps",
        ":stats_calculator_portable",
        "//third_party/eigen3",
        "@double_conversion//:double-conversion",
        "@farmhash_archive//:farmhash",
        "@nsync//:nsync_cpp",
        "@zlib_archive//:zlib",
    ],
    alwayslink = 1,
)

# Native library support for iOS applications.
#
# bazel  build --config=ios_x86_64 \
# :ios_tensorflow_lib
cc_library(
    name = "ios_tensorflow_lib",
    srcs = if_ios([
        ":android_op_registrations_and_gradients",
        "//tensorflow/core/kernels:android_core_ops",
        "//tensorflow/core/kernels:android_extended_ops",
    ]),
    copts = tf_copts() + ["-Os"],
    visibility = ["//visibility:public"],
    deps = [
        ":ios_tensorflow_lib_lite",
        ":protos_all_cc_impl",
        "//third_party/eigen3",
        "//third_party/fft2d:fft2d_headers",
        "@com_google_protobuf//:protobuf",
        "@fft2d",
        "@gemmlowp",
    ],
    alwayslink = 1,
)

cc_library(
    name = "ios_tensorflow_lib_lite",
    srcs = if_ios([":android_srcs"]),
    copts = tf_copts() + ["-Os"],
    visibility = ["//visibility:public"],
    deps = [
        ":mobile_additional_lib_deps",
        ":protos_all_cc_impl",
        ":stats_calculator_portable",
        "//third_party/eigen3",
        "@com_google_protobuf//:protobuf",
        "@double_conversion//:double-conversion",
        "@farmhash_archive//:farmhash",
        "@nsync//:nsync_cpp",
    ],
    alwayslink = 1,
)

cc_library(
    name = "ios_tensorflow_test_lib",
    testonly = 1,
    srcs = if_ios([":android_test_srcs"]),
    copts = tf_copts() + ["-Os"],
    tags = [
        "manual",
        "notap",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":android_test_proto_lib",
        ":ios_tensorflow_lib",
        "//tensorflow/core/platform/default/build_config:gtest",
        "//third_party/eigen3",
    ],
)

# Full TensorFlow library with operator support. Use this unless reducing
# binary size (by packaging a reduced operator set) is a concern.
cc_library(
    name = "android_tensorflow_lib",
    srcs = if_android([":android_op_registrations_and_gradients"]),
    copts = tf_copts(),
    tags = [
        "manual",
        "notap",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":android_tensorflow_lib_lite",
        ":protos_all_cc_impl",
        "//tensorflow/core/kernels:android_tensorflow_kernels",
        "//third_party/eigen3",
        "@com_google_protobuf//:protobuf",
    ],
    alwayslink = 1,
)

filegroup(
    name = "android_op_registrations_and_gradients",
    srcs = ["//tensorflow/c/kernels:android_all_ops"] + glob(
        [
            "ops/**/*.cc",
            "ops/**/*.h",
        ],
        exclude = [
            "**/*test.cc",
            "**/*testutil*",
            "**/*testlib*",
            "**/*main.cc",
            "**/tpu_*",
        ],
    ),
    visibility = ["//visibility:public"],
)

filegroup(
    name = "android_test_srcs",
    # TODO(andrewharp/nhua):
    # make more test-related sources portable e.g. "//tensorflow/core/platform:test.cc",
    srcs = [
        ":framework/fake_input.cc",
        ":framework/fake_input.h",
        ":framework/shape_inference_testutil.cc",
        ":framework/shape_inference_testutil.h",
        ":framework/tensor_testutil.cc",
        ":framework/tensor_testutil.h",
        ":util/reporter.cc",
        ":util/reporter.h",
        "//tensorflow/core/platform:test.h",
    ],
    visibility = ["//visibility:public"],
)

# This is like android_test_srcs, minus the things that are already in android_srcs.
filegroup(
    name = "android_test_srcs_no_core",
    srcs = [
        ":framework/shape_inference_testutil.cc",
        ":framework/shape_inference_testutil.h",
        ":framework/tensor_testutil.cc",
        ":framework/tensor_testutil.h",
        ":util/reporter.cc",
        ":util/reporter.h",
        "//tensorflow/core/platform:test.h",
    ],
    visibility = ["//visibility:public"],
)

# Portable library providing testing functionality for TensorFlow.
cc_library(
    name = "android_tensorflow_test_lib",
    testonly = 1,
    srcs = if_android([":android_test_srcs"]),
    hdrs = [
        "framework/fake_input.h",
        "framework/shape_inference_testutil.h",
        "framework/tensor_testutil.h",
        "util/reporter.h",
    ],
    copts = tf_copts(android_optimization_level_override = None),
    tags = [
        "manual",
        "notap",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":android_tensorflow_lib",
        ":protos_cc",
        "//tensorflow/core/platform/default/build_config:gtest",
        "//third_party/eigen3",
    ],
)

# -----------------------------------------------------------------------------
# Libraries with GPU facilities that are useful for writing kernels.
cc_library(
    name = "gpu_lib",
    srcs = [
        "common_runtime/gpu/gpu_event_mgr.cc",
    ],
    hdrs = [
        "common_runtime/gpu/gpu_event_mgr.h",
    ],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":framework",
        ":framework_internal",
        ":lib",
        ":lib_internal",
        ":protos_all_cc",
        ":stream_executor",
    ],
)

cc_library(
    name = "gpu_headers_lib",
    hdrs = [
        "common_runtime/gpu/gpu_event_mgr.h",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cuda",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core/platform/default/build_config:cuda",
    ],
)

cc_library(
    name = "rocm",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core/platform/default/build_config:rocm",
    ],
)

# -----------------------------------------------------------------------------
# Clif-related proto libraries.

tf_pyclif_proto_library(
    name = "example/example_pyclif",
    proto_lib = ":protos_all_cc",
    proto_srcfile = "example/example.proto",
    visibility = ["//visibility:public"],
)

tf_pyclif_proto_library(
    name = "example/feature_pyclif",
    proto_lib = ":protos_all_cc",
    proto_srcfile = "example/feature.proto",
    visibility = ["//visibility:public"],
)

tf_pyclif_proto_library(
    name = "framework/cost_graph_pyclif",
    proto_lib = ":protos_all_cc",
    proto_srcfile = "framework/cost_graph.proto",
    visibility = ["//visibility:public"],
)

tf_pyclif_proto_library(
    name = "framework/tensor_pyclif",
    proto_lib = ":protos_all_cc",
    proto_srcfile = "framework/tensor.proto",
    visibility = ["//visibility:public"],
)

tf_pyclif_proto_library(
    name = "framework/kernel_def_pyclif",
    proto_lib = ":protos_all_cc",
    proto_srcfile = "framework/kernel_def.proto",
    visibility = ["//visibility:public"],
)

tf_pyclif_proto_library(
    name = "framework/node_def_pyclif",
    proto_lib = ":protos_all_cc",
    proto_srcfile = "framework/node_def.proto",
    visibility = ["//visibility:public"],
)

tf_pyclif_proto_library(
    name = "framework/function_pyclif",
    proto_lib = ":protos_all_cc",
    proto_srcfile = "framework/function.proto",
    visibility = ["//visibility:public"],
)

tf_pyclif_proto_library(
    name = "framework/graph_pyclif",
    deprecated_proto_lib_visibility = True,  # TODO(b/136847937)
    proto_lib = ":protos_all_cc",
    proto_srcfile = "framework/graph.proto",
    visibility = ["//visibility:public"],
)

tf_pyclif_proto_library(
    name = "framework/step_stats_pyclif",
    proto_lib = ":protos_all_cc",
    proto_srcfile = "framework/step_stats.proto",
    visibility = ["//visibility:public"],
)

tf_pyclif_proto_library(
    name = "framework/types_pyclif",
    proto_lib = ":protos_all_cc",
    proto_srcfile = "framework/types.proto",
    visibility = ["//visibility:public"],
)

tf_pyclif_proto_library(
    name = "protobuf/config_pyclif",
    proto_lib = ":protos_all_cc",
    proto_srcfile = "protobuf/config.proto",
    visibility = ["//visibility:public"],
)

tf_pyclif_proto_library(
    name = "protobuf/device_properties_pyclif",
    proto_lib = ":protos_all_cc",
    proto_srcfile = "protobuf/device_properties.proto",
    visibility = ["//visibility:public"],
)

tf_pyclif_proto_library(
    name = "protobuf/graph_debug_info_pyclif",
    proto_lib = ":protos_all_cc",
    proto_srcfile = "protobuf/graph_debug_info.proto",
    visibility = ["//visibility:public"],
)

tf_pyclif_proto_library(
    name = "protobuf/meta_graph_pyclif",
    proto_lib = ":protos_all_cc",
    proto_srcfile = "protobuf/meta_graph.proto",
    visibility = ["//visibility:public"],
)

tf_pyclif_proto_library(
    name = "protobuf/saved_model_pyclif",
    proto_lib = ":protos_all_cc",
    proto_srcfile = "protobuf/saved_model.proto",
    visibility = ["//visibility:public"],
)

tf_pyclif_proto_library(
    name = "framework/variable_pyclif",
    proto_lib = ":protos_all_cc",
    proto_srcfile = "framework/variable.proto",
    visibility = ["//visibility:public"],
)

# -----------------------------------------------------------------------------
# Internal targets

tf_proto_library(
    name = "autotuning_proto",
    srcs = ["protobuf/autotuning.proto"],
    cc_api_version = 2,
    make_default_target_header_only = True,
    visibility = [
        "//tensorflow:internal",
    ],
)

tf_proto_library(
    name = "conv_autotuning_proto",
    srcs = ["protobuf/conv_autotuning.proto"],
    cc_api_version = 2,
    make_default_target_header_only = True,
    protodeps = [
        "//tensorflow/stream_executor:dnn_proto",
    ],
    visibility = [
        "//tensorflow:internal",
    ],
)

tf_proto_library_cc(
    name = "worker_proto",
    srcs = ["protobuf/worker.proto"],
    cc_api_version = 2,
    protodeps = tf_additional_all_protos(),
    visibility = ["//visibility:public"],
)

tf_proto_library_cc(
    name = "worker_service_proto",
    srcs = ["protobuf/worker_service.proto"],
    has_services = 1,
    cc_api_version = 2,
    cc_stubby_versions = ["2"],
    protodeps = [":worker_proto"],
    visibility = [
        "//tensorflow:internal",
    ],
)

tf_proto_library_cc(
    name = "master_proto",
    srcs = ["protobuf/master.proto"],
    cc_api_version = 2,
    protodeps = tf_additional_all_protos(),
    visibility = ["//tensorflow:internal"],
)

tf_proto_library_cc(
    name = "master_service_proto",
    srcs = ["protobuf/master_service.proto"],
    has_services = 1,
    cc_api_version = 2,
    cc_stubby_versions = ["2"],
    protodeps = [":master_proto"],
    visibility = [
        "//tensorflow:internal",
    ],
)

tf_proto_library_cc(
    name = "eager_service_proto",
    srcs = ["protobuf/eager_service.proto"],
    has_services = 1,
    cc_api_version = 2,
    cc_grpc_version = 1,
    cc_stubby_versions = ["2"],
    protodeps = tf_additional_all_protos(),
    visibility = [
        "//tensorflow:internal",
    ],
)

LIB_INTERNAL_PRIVATE_HEADERS = [
    "framework/resource_handle.h",
    "//tensorflow/core/platform:legacy_lib_internal_headers",
    "//tensorflow/core/platform:scanner.h",
    "//tensorflow/core/platform:str_util.h",
    "//tensorflow/core/lib/bfloat16:bfloat16.h",
    "//tensorflow/core/lib/core:legacy_lib_core_all_headers",
    "//tensorflow/core/lib/gtl:legacy_lib_gtl_all_headers",
    "//tensorflow/core/lib/hash:legacy_lib_hash_all_headers",
    "//tensorflow/core/lib/io:legacy_lib_io_all_headers",
    "//tensorflow/core/lib/random:legacy_lib_random_all_headers",
    "//tensorflow/core/lib/strings:legacy_lib_strings_all_headers",
    "//tensorflow/core/lib/math:math_util.h",
] + glob(
    [
        "lib/**/*.h",
    ],
    exclude = [
        "**/*test*",
        "lib/gif/**/*",
        "lib/jpeg/**/*",
        "lib/png/**/*",
    ],
)

LIB_INTERNAL_PUBLIC_HEADERS = [
    "//tensorflow/core/lib/core:legacy_lib_internal_core_headers",
    "//tensorflow/core/lib/gtl:legacy_lib_internal_public_gtl_headers",
    "//tensorflow/core/lib/hash:legacy_lib_internal_public_headers",
    "//tensorflow/core/lib/io:legacy_lib_internal_public_headers",
    "lib/monitoring/mobile_counter.h",
    "lib/monitoring/mobile_gauge.h",
    "lib/monitoring/mobile_sampler.h",
    "lib/png/png_io.h",
    "//tensorflow/core/lib/random:legacy_lib_internal_public_random_headers",
    "//tensorflow/core/lib/strings:legacy_lib_internal_public_string_headers",
    "lib/wav/wav_io.h",
    "//tensorflow/core/platform:demangle.h",
    "//tensorflow/core/platform:denormal.h",
    "//tensorflow/core/platform:host_info.h",
    "//tensorflow/core/platform:platform.h",
    "//tensorflow/core/platform:monitoring.h",
    "//tensorflow/core/platform:protobuf_internal.h",
    "//tensorflow/core/platform:setround.h",
    "//tensorflow/core/platform:snappy.h",
    "//tensorflow/core/platform:tensor_coding.h",
    "//tensorflow/core/platform:tracing.h",
    "//tensorflow/core/platform:unbounded_work_queue.h",
    "//tensorflow/core/platform:legacy_platform_lib_hdrs",
    "util/env_var.h",
]

# Replicated for lib_internal and lib_internal_impl.
LIB_INTERNAL_DEFINES = (
    tf_additional_lib_defines() + [
        "TF_USE_SNAPPY",
    ] + tf_additional_numa_lib_defines()
)

cc_library(
    name = "lib_internal",
    srcs = LIB_INTERNAL_PRIVATE_HEADERS,
    hdrs = LIB_INTERNAL_PUBLIC_HEADERS,
    copts = tf_copts(),
    defines = LIB_INTERNAL_DEFINES,
    linkopts = select({
        "//tensorflow:freebsd": [],
        "//tensorflow:windows": [],
        "//tensorflow:android": [],
        "//conditions:default": [
            "-ldl",
            "-lpthread",
        ],
    }),
    deps = tf_additional_lib_deps() + [
        "//tensorflow/core/platform:annotation",
        "@com_google_absl//absl/meta:type_traits",
        "@com_google_absl//absl/strings",
        "//third_party/eigen3",
        "@com_google_absl//absl/base:core_headers",
        "//tensorflow/core/platform/default/build_config:platformlib",
    ] + if_static([":lib_internal_impl"]),
)

cc_library(
    name = "lib_internal_impl",
    srcs = LIB_INTERNAL_PRIVATE_HEADERS + glob(
        [
            "lib/**/*.cc",
            "util/env_var.cc",
        ],
        exclude = [
            "**/*test*",
            "framework/variant.cc",
            "lib/gif/**/*",
            "lib/jpeg/**/*",
            "lib/png/**/*",
        ],
    ) + [
        "//tensorflow/core/platform:legacy_monitoring_srcs",
        "//tensorflow/core/platform:legacy_platform_lib_srcs",
        "//tensorflow/core/platform:legacy_lib_internal_srcs",
        "//tensorflow/core/lib/core:legacy_lib_core_all_srcs",
        "//tensorflow/core/lib/hash:legacy_lib_internal_impl_srcs",
        "//tensorflow/core/lib/io:legacy_lib_io_all_srcs",
        "//tensorflow/core/lib/random:legacy_lib_random_all_srcs",
        "//tensorflow/core/lib/strings:legacy_lib_strings_all_srcs",
    ],
    hdrs = LIB_INTERNAL_PUBLIC_HEADERS,
    copts = tf_copts(),
    defines = LIB_INTERNAL_DEFINES,
    deps = tf_additional_lib_deps() + [
               ":core_stringpiece",
               ":lib_hash_crc32c_accelerate_internal",
               ":lib_proto_parsing",
               "@com_google_absl//absl/memory",
               "@com_google_absl//absl/strings",
               "//third_party/eigen3",
               "//tensorflow/core/lib/bfloat16",
               "//tensorflow/core/lib/math:math_util",
               "//tensorflow/core/platform:abi",
               "//tensorflow/core/platform:annotation",
               "//tensorflow/core/platform:cpu_info",
               "//tensorflow/core/platform:denormal",
               "//tensorflow/core/platform:dynamic_annotations",
               "//tensorflow/core/platform:mutex",
               "//tensorflow/core/platform:notification",
               "//tensorflow/core/platform:numbers",
               "//tensorflow/core/platform:platform_strings",
               "//tensorflow/core/platform:scanner",
               "//tensorflow/core/platform:setround",
               "//tensorflow/core/platform:strong_hash",
               "//tensorflow/core/platform:strcat",
               "//tensorflow/core/platform:stringprintf",
               "//tensorflow/core/platform:str_util",
               "//tensorflow/core/platform:thread_annotations",
               "//tensorflow/core/platform/default/build_config:platformlib",
               "@snappy",
               "@zlib_archive//:zlib",
               "@double_conversion//:double-conversion",
               "@com_google_protobuf//:protobuf",
           ] + tf_protos_all_impl() + tf_protos_grappler_impl() +
           tf_additional_numa_deps(),
    alwayslink = 1,
)

# File compiled with extra flags to get cpu-specific acceleration.
cc_library(
    name = "lib_hash_crc32c_accelerate_internal",
    srcs = [
        "//tensorflow/core/lib/hash:legacy_crc32_accelerate_srcs",
    ],
    # -msse4.2 enables the use of crc32c compiler builtins.
    copts = tf_copts() + if_linux_x86_64(["-msse4.2"]),
)

cc_library(
    name = "gif_internal",
    srcs = [
        "lib/gif/gif_io.cc",
        "//tensorflow/core/platform:gif.h",
    ],
    hdrs = ["lib/gif/gif_io.h"],
    copts = tf_copts(),
    linkopts = select({
        "//tensorflow:freebsd": [],
        "//tensorflow:windows": [],
        "//conditions:default": ["-ldl"],
    }),
    deps = [
        ":lib",
        ":lib_internal",
        "//tensorflow/core/platform:gif",
    ],
)

cc_library(
    name = "jpeg_internal",
    srcs = [
        "lib/jpeg/jpeg_handle.cc",
        "lib/jpeg/jpeg_mem.cc",
        "//tensorflow/core/platform:jpeg.h",
    ],
    hdrs = [
        "lib/jpeg/jpeg_handle.h",
        "lib/jpeg/jpeg_mem.h",
    ],
    copts = tf_copts(),
    linkopts = select({
        "//tensorflow:freebsd": [],
        "//tensorflow:windows": [],
        "//conditions:default": ["-ldl"],
    }),
    deps = [
        ":lib",
        ":lib_internal",
        "//tensorflow/core/platform:jpeg",
    ],
)

cc_library(
    name = "png_internal",
    srcs = ["lib/png/png_io.cc"],
    hdrs = [
        "lib/png/png_io.h",
        "//tensorflow/core/lib/bfloat16:bfloat16.h",
        "//tensorflow/core/lib/core:legacy_lib_core_stringpiece_header",
        "//tensorflow/core/platform:byte_order.h",
        "//tensorflow/core/platform:cpu_info.h",
        "//tensorflow/core/platform:default/integral_types.h",
        "//tensorflow/core/platform:default/logging.h",
        "//tensorflow/core/platform:logging.h",
        "//tensorflow/core/platform:macros.h",
        "//tensorflow/core/platform:platform.h",
        "//tensorflow/core/platform:png.h",
        "//tensorflow/core/platform:tstring.h",
        "//tensorflow/core/platform:types.h",
    ],
    copts = tf_copts(),
    linkopts = select({
        "//tensorflow:freebsd": [],
        "//tensorflow:windows": [],
        "//conditions:default": ["-ldl"],
    }),
    deps = [
        ":lib",
        ":lib_internal",
        "//tensorflow/core/platform:png",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/strings",
        "@zlib_archive//:zlib",
    ],
)

cc_library(
    name = "tflite_portable_logging",
    hdrs = [
        "//tensorflow/core/lib/bfloat16:bfloat16.h",
        "//tensorflow/core/platform:default/integral_types.h",
        "//tensorflow/core/platform:default/logging.h",
        "//tensorflow/core/platform:logging.h",
        "//tensorflow/core/platform:macros.h",
        "//tensorflow/core/platform:platform.h",
        "//tensorflow/core/platform:tstring.h",
        "//tensorflow/core/platform:types.h",
    ],
    copts = tf_copts(),
    linkopts = ["-ldl"],
    deps = [
        ":platform_base",
        "//tensorflow/core/platform/default/build_config:logging",
    ],
)

cc_library(
    name = "android_jpeg_internal",
    srcs = if_android([
        "lib/jpeg/jpeg_handle.cc",
        "lib/jpeg/jpeg_mem.cc",
        "//tensorflow/core/platform:jpeg.h",
    ]),
    hdrs = [
        "lib/jpeg/jpeg_handle.h",
        "lib/jpeg/jpeg_mem.h",
        "//tensorflow/core/lib/bfloat16:bfloat16.h",
        "//tensorflow/core/lib/core:legacy_lib_core_stringpiece_header",
        "//tensorflow/core/platform:default/integral_types.h",
        "//tensorflow/core/platform:default/logging.h",
        "//tensorflow/core/platform:dynamic_annotations.h",
        "//tensorflow/core/platform:logging.h",
        "//tensorflow/core/platform:macros.h",
        "//tensorflow/core/platform:mem.h",
        "//tensorflow/core/platform:platform.h",
        "//tensorflow/core/platform:stringpiece.h",
        "//tensorflow/core/platform:tstring.h",
        "//tensorflow/core/platform:types.h",
    ],
    copts = tf_copts(),
    linkopts = ["-ldl"],
    deps = [
        ":core_stringpiece",
        "//tensorflow/core/platform:dynamic_annotations",
        "//tensorflow/core/platform:jpeg",
        "//tensorflow/core/platform:stringpiece",
        "//tensorflow/core/platform/default/build_config:logging",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "android_gif_internal",
    srcs = if_android([
        "lib/gif/gif_io.cc",
        "//tensorflow/core/platform:gif.h",
        "//tensorflow/core/lib/strings:legacy_lib_android_gif_internal_string_headers",
    ]),
    hdrs = [
        "lib/gif/gif_io.h",
        "//tensorflow/core/lib/bfloat16:bfloat16.h",
        "//tensorflow/core/lib/core:legacy_lib_core_stringpiece_header",
        "//tensorflow/core/lib/gtl:legacy_android_gif_internal_headers",
        "//tensorflow/core/platform:default/integral_types.h",
        "//tensorflow/core/platform:default/logging.h",
        "//tensorflow/core/platform:dynamic_annotations.h",
        "//tensorflow/core/platform:logging.h",
        "//tensorflow/core/platform:macros.h",
        "//tensorflow/core/platform:mem.h",
        "//tensorflow/core/platform:platform.h",
        "//tensorflow/core/platform:tstring.h",
        "//tensorflow/core/platform:types.h",
    ],
    copts = tf_copts(),
    linkopts = ["-ldl"],
    deps = [
        "//tensorflow/core/platform:dynamic_annotations",
        "//tensorflow/core/platform:gif",
        "//tensorflow/core/platform:numbers",
        "//tensorflow/core/platform:strcat",
        "//tensorflow/core/platform:stringpiece",
        "//tensorflow/core/platform/default/build_config:logging",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "android_png_internal",
    srcs = if_android([
        "lib/png/png_io.cc",
        "//tensorflow/core/platform:png.h",
    ]),
    hdrs = [
        "lib/png/png_io.h",
        "//tensorflow/core/lib/bfloat16:bfloat16.h",
        "//tensorflow/core/lib/core:legacy_lib_core_stringpiece_header",
        "//tensorflow/core/platform:byte_order.h",
        "//tensorflow/core/platform:cpu_info.h",
        "//tensorflow/core/platform:default/integral_types.h",
        "//tensorflow/core/platform:default/logging.h",
        "//tensorflow/core/platform:logging.h",
        "//tensorflow/core/platform:macros.h",
        "//tensorflow/core/platform:platform.h",
        "//tensorflow/core/platform:tstring.h",
        "//tensorflow/core/platform:types.h",
    ],
    copts = tf_copts(),
    linkopts = ["-ldl"],
    deps = [
        "//tensorflow/core/platform:stringpiece",
        "//tensorflow/core/platform/default/build_config:logging",
        "@com_google_absl//absl/strings",
        "@png",
    ],
)

tf_proto_library(
    name = "error_codes_proto_impl",
    srcs = ["protobuf/error_codes.proto"],
    cc_api_version = 2,
    make_default_target_header_only = True,
)

tf_proto_library(
    name = "protos_all_proto",
    srcs = COMMON_PROTO_SRCS + ADDITIONAL_CORE_PROTO_SRCS,
    cc_api_version = 2,
    make_default_target_header_only = True,
    protodeps = [
        ":error_codes_proto_impl",
        "//tensorflow/core/lib/core:error_codes_proto",
    ],
)

alias(
    name = "error_codes_proto_cc",
    actual = "//tensorflow/core/lib/core:error_codes_proto_cc",
)

tf_version_info_genrule()

cc_library(
    name = "version_lib",
    srcs = ["util/version_info.cc"],
    hdrs = ["public/version.h"],
    copts = tf_copts(),
)

FRAMEWORK_INTERNAL_PRIVATE_HEADERS = [
    "graph/edgeset.h",
    "graph/graph.h",
    "graph/graph_def_builder.h",
    "graph/node_builder.h",
    "graph/tensor_id.h",
] + glob(
    [
        "example/**/*.h",
        "framework/**/*.h",
        "util/**/*.h",
    ],
    exclude = [
        "**/*test*",
        "**/*main.cc",
        "example/example_parser_configuration.*",
        "util/reporter.h",
        "util/reporter.cc",
        "util/port.h",
        "framework/fake_input.*",
        "framework/op_gen_lib.*",
        "framework/reader_base.*",
        "util/memmapped_file_system.*",
        "util/memmapped_file_system_writer.*",
        "util/session_message.*",
        "util/version_info.cc",
    ],
) + select({
    "//tensorflow:windows": [],
    "//conditions:default": [
        "util/memmapped_file_system.h",
        "util/memmapped_file_system_writer.h",
    ],
})

FRAMEWORK_INTERNAL_PUBLIC_HEADERS = [
    "framework/model.h",  # only needed for tests
    "framework/op_segment.h",
    "framework/rendezvous.h",  # only needed for tests
    "framework/resource_var.h",
    "framework/run_handler.h",
    "framework/run_handler_util.h",
    "framework/shared_ptr_variant.h",
    "framework/tensor_reference.h",
    "framework/tracking_allocator.h",  # only needed for tests
    "framework/unique_tensor_references.h",
    "framework/variant.h",
    "util/command_line_flags.h",
    "util/equal_graph_def.h",
    "util/presized_cuckoo_map.h",
    "util/tensor_slice_set.h",
    "util/tensor_slice_util.h",
    "util/xla_config_registry.h",
]

tf_cuda_library(
    name = "framework_internal",
    srcs = FRAMEWORK_INTERNAL_PRIVATE_HEADERS,
    hdrs = FRAMEWORK_INTERNAL_PUBLIC_HEADERS,
    deps = [
        ":framework_internal_headers_lib",
        "//third_party/eigen3",
        ":lib",
    ] + if_static(
        extra_deps = [
            ":framework_internal_impl",
            "@com_google_protobuf//:protobuf",
        ],
        otherwise = [
            "@com_google_protobuf//:protobuf_headers",
        ],
    ),
    alwayslink = 1,
)

cc_header_only_library(
    name = "framework_internal_headers_lib",
    # Fully depend on external repositories, because identifying the headers
    # is fragile.
    extra_deps = [
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
    deps = [
        ":lib",
        ":lib_internal",
        ":version_lib",
        "//tensorflow/core/kernels:bounds_check",
        "//tensorflow/core/platform/default/build_config:platformlib",
    ],
)

cc_header_only_library(
    name = "core_cpu_headers_lib",
    visibility = ["//visibility:public"],
    deps = [
        ":core_cpu_lib",
    ],
)

tf_cuda_library(
    name = "framework_internal_impl",
    srcs = FRAMEWORK_INTERNAL_PRIVATE_HEADERS + glob(
        [
            "example/**/*.cc",
            "framework/**/*.cc",
            "util/**/*.cc",
            "graph/edgeset.cc",
            "graph/graph.cc",
            "graph/graph_def_builder.cc",
            "graph/node_builder.cc",
            "graph/tensor_id.cc",
            "graph/while_context.h",
            "graph/while_context.cc",
        ],
        exclude = [
            "**/*test*",
            "**/*main.cc",
            "framework/allocator.cc",
            "framework/cpu_allocator_impl.cc",
            "framework/allocator_registry.cc",
            "framework/tracking_allocator.cc",
            "example/example_parser_configuration.*",
            "example/feature_util.cc",
            "util/reporter.cc",
            "framework/fake_input.*",
            "framework/op_gen_lib.*",
            "framework/reader_base.*",
            "util/memmapped_file_system.*",
            "util/memmapped_file_system_writer.*",
            "util/stats_calculator.*",
            "util/version_info.cc",
            "util/env_var.cc",
            "util/port.cc",
        ],
    ) + select({
        "//tensorflow:windows": [],
        "//conditions:default": [
            "util/memmapped_file_system.cc",
            "util/memmapped_file_system_writer.cc",
        ],
    }),
    hdrs = FRAMEWORK_INTERNAL_PUBLIC_HEADERS,
    copts = tf_copts(),
    linkopts = select({
        "//tensorflow:freebsd": ["-lm"],
        "//tensorflow:windows": [],
        "//conditions:default": [
            "-ldl",
            "-lm",
        ],
    }),
    deps = [
        ":allocator_registry_impl",
        ":allocator",
        ":attr_value_proto_text",
        ":feature_util",
        ":lib",
        ":lib_internal",
        ":protos_all_cc",
        ":stats_calculator_portable",
        ":version_lib",
        ":util_port",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "//tensorflow/core/platform/default/build_config:platformlib",
        "//tensorflow/core/kernels:bounds_check",
        "//tensorflow/core/profiler/lib:traceme",
        "//third_party/eigen3",
    ] + if_static(
        extra_deps = ["@com_google_protobuf//:protobuf"],
        otherwise = ["@com_google_protobuf//:protobuf_headers"],
    ) + mkl_deps(),
    alwayslink = 1,
)

cc_header_only_library(
    name = "framework_headers_lib",
    # Fully depend on external repositories, because identifying the headers
    # is fragile.
    extra_deps = [
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":framework",
        ":reader_base",
    ],
)

cc_header_only_library(
    name = "stream_executor_headers_lib",
    # Fully depend on external repositories, because identifying the headers
    # is fragile.
    extra_deps = [
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":stream_executor",
    ],
)

tf_cuda_library(
    name = "stream_executor",
    srcs = ["//tensorflow/core/platform:stream_executor.h"],
    hdrs = [
        "//tensorflow/core/platform:cuda.h",
        "//tensorflow/core/platform:rocm.h",
        "//tensorflow/core/platform:stream_executor.h",
    ],
    deps = [
        "//tensorflow/core/platform/default/build_config:stream_executor",
    ],
)

# Like stream_executor library, but compiles without --config=cuda
# and does not include any cuda dependencies.
cc_library(
    name = "stream_executor_no_cuda",
    srcs = ["//tensorflow/core/platform:stream_executor.h"],
    hdrs = [
        "//tensorflow/core/platform:stream_executor_no_cuda.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core/platform/default/build_config:stream_executor_no_cuda",
    ],
)

tf_cuda_library(
    name = "cuda_device_functions",
    hdrs = [
        "util/gpu_device_functions.h",
    ],
    visibility = ["//visibility:public"],
    deps = [":framework_lite"],
)

# TODO(josh11b): Is this needed, or can we just use ":protos_all_cc"?
cc_library(
    name = "protos_cc",
    visibility = ["//visibility:public"],
    deps = ["//tensorflow/core/platform/default/build_config:protos_cc"],
)

# Library containing all of the graph construction code that is
# independent of the runtime.
#
# TODO(mrry): Refactor graph_constructor.cc so that it does not depend on code
# in "common_runtime/", and then the entire "graph/" directory can be included
# in this library.
GRAPH_HDRS = [
    "graph/algorithm.h",
    "graph/collective_order.h",
    "graph/colors.h",
    "graph/control_flow.h",
    "graph/costmodel.h",
    "graph/default_device.h",
    "graph/edgeset.h",
    "graph/graph.h",
    "graph/graph_constructor.h",  # NOTE(mrry): Don't include the .cc since it depends on common_runtime.
    "graph/graph_def_builder.h",
    "graph/graph_def_builder_util.h",
    "graph/graph_partition.h",
    "graph/mkl_layout_pass.h",
    "graph/mkl_tfconversion_pass.h",
    "graph/node_builder.h",
    "graph/optimizer_cse.h",
    "graph/subgraph.h",
    "graph/tensor_id.h",
    "graph/testlib.h",
    "graph/types.h",
    "graph/validate.h",
    "graph/while_context.h",
]

tf_cuda_library(
    name = "graph",
    srcs = [
        "graph/algorithm.cc",
        "graph/collective_order.cc",
        "graph/colors.cc",
        "graph/control_flow.cc",
        "graph/costmodel.cc",
        "graph/graph_partition.cc",
        "graph/optimizer_cse.cc",
        "graph/subgraph.cc",
        "graph/validate.cc",
    ],
    hdrs = GRAPH_HDRS,
    deps = [
        ":framework",
        ":framework_internal",
        ":lib",
        ":lib_internal",
        ":protos_all_cc",
        "//third_party/eigen3",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

CORE_CPU_BASE_HDRS = GRAPH_HDRS + [
    "common_runtime/device.h",
    "common_runtime/device_factory.h",
    "common_runtime/device_mgr.h",
    "common_runtime/device_set.h",
    "common_runtime/eval_const_tensor.h",
    "common_runtime/graph_runner.h",
    "common_runtime/shape_refiner.h",
    "framework/versions.h",
    "common_runtime/process_function_library_runtime.h",
    "common_runtime/function.h",
    "common_runtime/scoped_allocator.h",
    "common_runtime/scoped_allocator_mgr.h",
]

tf_cuda_library(
    name = "core_cpu_base",
    hdrs = CORE_CPU_BASE_HDRS + ["public/session.h"],
    copts = tf_copts(),
    deps = [":core_cpu_base_no_ops"] + if_static([
        ":function_ops_op_lib",
        ":functional_grad",
        ":functional_ops_op_lib",
        "//tensorflow/core/kernels:bounds_check",
        "//tensorflow/core/kernels:required",
    ]),
    alwayslink = 1,
)

tf_cuda_library(
    name = "core_cpu_base_no_ops",
    srcs = [
        "common_runtime/eval_const_tensor.cc",
        "common_runtime/scoped_allocator.cc",
        "common_runtime/scoped_allocator_mgr.cc",
        "common_runtime/shape_refiner.cc",
        "common_runtime/graph_optimizer.h",
        "graph/graph_constructor.cc",  # Depends on common_runtime.
        "graph/graph_def_builder_util.cc",  # Depends on common_runtime.
        "public/session_options.h",
        "public/version.h",
    ] + CORE_CPU_BASE_HDRS,
    hdrs = CORE_CPU_BASE_HDRS + ["public/session.h"],
    copts = tf_copts(),
    deps = [
        ":graph",
        ":framework",
        ":framework_internal",
        ":lib",
        ":lib_internal",
        ":protos_all_cc",
        "@com_google_absl//absl/container:flat_hash_set",
        "//third_party/eigen3",
    ] + if_static([
        "@com_google_absl//absl/algorithm:container",
    ]),
)

CORE_CPU_LIB_HEADERS = CORE_CPU_BASE_HDRS + [
    "common_runtime/allocator_retry.h",
    "common_runtime/shared_counter.h",
    "common_runtime/base_collective_executor.h",
    "common_runtime/bfc_allocator.h",
    "common_runtime/hierarchical_tree_broadcaster.h",
    "common_runtime/buf_rendezvous.h",
    "common_runtime/build_graph_options.h",
    "common_runtime/collective_executor_mgr.h",
    "common_runtime/collective_param_resolver_local.h",
    "common_runtime/collective_rma_local.h",
    "common_runtime/collective_util.h",
    "common_runtime/colocation_graph.h",
    "common_runtime/constant_folding.h",
    "common_runtime/copy_tensor.h",
    "common_runtime/costmodel_manager.h",
    "common_runtime/placer_inspection_required_ops_utils.h",
    "common_runtime/debugger_state_interface.h",
    "common_runtime/device_resolver_local.h",
    "common_runtime/dma_helper.h",
    "common_runtime/executor.h",
    "common_runtime/executor_factory.h",
    "common_runtime/graph_optimizer.h",
    "common_runtime/input_colocation_exemption_registry.h",
    "common_runtime/isolate_placer_inspection_required_ops_pass.h",
    "common_runtime/local_device.h",
    "common_runtime/lower_function_call_op.h",
    "common_runtime/lower_if_op.h",
    "common_runtime/lower_case_op.h",
    "common_runtime/lower_functional_ops.h",
    "common_runtime/lower_while_op.h",
    "common_runtime/memory_types.h",
    "common_runtime/metrics.h",
    "common_runtime/mkl_cpu_allocator.h",
    "common_runtime/optimization_registry.h",
    "common_runtime/pending_counts.h",
    "common_runtime/partitioning_utils.h",
    "common_runtime/placer.h",
    "common_runtime/process_util.h",
    "common_runtime/inspecting_placer.h",
    "common_runtime/profile_handler.h",
    "common_runtime/renamed_device.h",
    "common_runtime/rendezvous_mgr.h",
    "common_runtime/rendezvous_util.h",
    "common_runtime/ring_reducer.h",
    "common_runtime/ring_alg.h",
    "common_runtime/ring_gatherer.h",
    "common_runtime/session_factory.h",
    "common_runtime/single_threaded_cpu_device.h",
    "common_runtime/stats_publisher_interface.h",
    "common_runtime/step_stats_collector.h",
    "common_runtime/threadpool_device.h",
    "common_runtime/process_state.h",
    "common_runtime/pool_allocator.h",
    "graph/gradients.h",
    "graph/quantize_training.h",
] + if_mkl(["graph/mkl_graph_util.h"])

tf_cuda_library(
    name = "core_cpu_impl",
    srcs = [
        "common_runtime/accumulate_n_optimizer.cc",
        "common_runtime/base_collective_executor.cc",
        "common_runtime/buf_rendezvous.cc",
        "common_runtime/build_graph_options.cc",
        "common_runtime/collective_executor_mgr.cc",
        "common_runtime/collective_param_resolver_local.cc",
        "common_runtime/collective_rma_local.cc",
        "common_runtime/collective_util.cc",
        "common_runtime/colocation_graph.cc",
        "common_runtime/constant_folding.cc",
        "common_runtime/copy_tensor.cc",
        "common_runtime/costmodel_manager.cc",
        "common_runtime/debugger_state_interface.cc",
        "common_runtime/device.cc",
        "common_runtime/device_factory.cc",
        "common_runtime/device_mgr.cc",
        "common_runtime/device_resolver_local.cc",
        "common_runtime/device_set.cc",
        "common_runtime/dynamic_device_mgr.cc",
        "common_runtime/executor.cc",
        "common_runtime/executor_factory.cc",
        "common_runtime/function.cc",
        "common_runtime/graph_optimizer.cc",
        "common_runtime/graph_runner.cc",
        "common_runtime/hierarchical_tree_broadcaster.cc",
        "common_runtime/input_colocation_exemption_registry.cc",
        "common_runtime/inspecting_placer.cc",
        "common_runtime/isolate_placer_inspection_required_ops_pass.cc",
        "common_runtime/local_device.cc",
        "common_runtime/lower_case_op.cc",
        "common_runtime/lower_function_call_op.cc",
        "common_runtime/lower_functional_ops.cc",
        "common_runtime/lower_if_op.cc",
        "common_runtime/lower_while_op.cc",
        "common_runtime/memory_types.cc",
        "common_runtime/metrics.cc",
        "common_runtime/mkl_cpu_allocator.cc",
        "common_runtime/optimization_registry.cc",
        "common_runtime/parallel_concat_optimizer.cc",
        "common_runtime/partitioning_utils.cc",
        "common_runtime/placer.cc",
        "common_runtime/placer_inspection_required_ops_utils.cc",
        "common_runtime/placer_inspection_required_ops_utils.h",
        "common_runtime/pool_allocator.cc",
        "common_runtime/process_function_library_runtime.cc",
        "common_runtime/process_state.cc",
        "common_runtime/process_util.cc",
        "common_runtime/renamed_device.cc",
        "common_runtime/rendezvous_mgr.cc",
        "common_runtime/rendezvous_util.cc",
        "common_runtime/ring_alg.cc",
        "common_runtime/ring_gatherer.cc",
        "common_runtime/ring_reducer.cc",
        "common_runtime/session.cc",
        "common_runtime/session_factory.cc",
        "common_runtime/session_options.cc",
        "common_runtime/session_state.cc",
        "common_runtime/single_threaded_cpu_device.cc",
        "common_runtime/stats_publisher_interface.cc",
        "common_runtime/step_stats_collector.cc",
        "common_runtime/threadpool_device.cc",
        "common_runtime/threadpool_device_factory.cc",
        "graph/gradients.cc",
        "graph/mkl_layout_pass.cc",
        "graph/mkl_tfconversion_pass.cc",
        "graph/quantize_training.cc",
        "public/session.h",
        "public/session_options.h",
        "public/version.h",
    ],
    hdrs = CORE_CPU_LIB_HEADERS,
    copts = tf_copts() + tf_openmp_copts(),
    deps = [
        ":bfc_allocator",
        ":graph",
        ":framework",
        ":framework_internal",
        ":lib",
        ":lib_internal",
        ":protos_all_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "//third_party/eigen3",
        "//tensorflow/core/grappler/utils:functions",
        "//tensorflow/core/profiler/lib:traceme",
        "//tensorflow/core/profiler/internal:traceme_recorder",
    ] + mkl_deps(),
    alwayslink = 1,
)

tf_cuda_library(
    name = "core_cpu_lib",
    hdrs = CORE_CPU_LIB_HEADERS,
    deps = [
        ":core_cpu_base",
        "//tensorflow/core/grappler:grappler_item",
    ] + if_static([":core_cpu_impl"]) + tf_protos_all() + tf_protos_grappler(),
)

tf_cuda_library(
    name = "core_cpu_lib_no_ops",
    hdrs = CORE_CPU_LIB_HEADERS,
    deps = [
        ":core_cpu_base_no_ops",
        "//tensorflow/core/grappler:grappler_item",
    ] + tf_protos_all() + tf_protos_grappler(),
)

tf_cuda_library(
    name = "core_cpu_internal",
    srcs = [
        "common_runtime/graph_execution_state.cc",
    ],
    hdrs = [
        "common_runtime/graph_execution_state.h",
    ] + CORE_CPU_LIB_HEADERS,
    copts = tf_copts(),
    deps = [
        ":framework",
        ":graph",
        ":lib",
        ":protos_all_cc",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler/clusters:utils",
        "//tensorflow/core/grappler/clusters:virtual_cluster",
        "//tensorflow/core/grappler/optimizers:meta_optimizer",
        "//third_party/eigen3",
    ] + mkl_deps() + tf_additional_core_deps() + if_static([
        ":core_cpu_impl",
        ":function_ops_op_lib",
        ":functional_grad",
        ":functional_ops_op_lib",
        "//tensorflow/core/kernels:required",
    ]),
    alwayslink = 1,
)

# This is redundant with the "core_cpu_*" targets above. It's useful for
# applications that want to depend on a minimal subset of TensorFlow (e.g. XLA).
cc_library(
    name = "bfc_allocator",
    srcs = [
        "common_runtime/allocator_retry.cc",
        "common_runtime/allocator_retry.h",
        "common_runtime/bfc_allocator.cc",
    ],
    hdrs = ["common_runtime/bfc_allocator.h"],
    features = ["parse_headers"],
    visibility = ["//visibility:public"],
    deps = [
        ":allocator",
        ":lib",
        ":lib_internal",
        ":protos_all_cc",
        ":shared_counter",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

cc_library(
    name = "shared_counter",
    hdrs = ["common_runtime/shared_counter.h"],
    features = ["parse_headers"],
    visibility = ["//visibility:public"],
    deps = [
        ":lib",
    ],
)

cc_library(
    name = "regexp_internal",
    hdrs = [
        "//tensorflow/core/platform:regexp.h",
    ],
    visibility = [
        "//tensorflow/compiler:__subpackages__",
        "//tensorflow/core/kernels:__subpackages__",
        "//tensorflow/core/profiler:__subpackages__",
        "//tensorflow/stream_executor:__subpackages__",
    ],
    deps = [":lib_internal"],
)

tf_cuda_library(
    name = "direct_session_internal",
    srcs = ["common_runtime/direct_session.cc"],
    hdrs = [
        "common_runtime/direct_session.h",
        "util/env_var.h",
    ],
    copts = tf_copts(),
    deps = [
        ":core_cpu_internal",
        ":framework",
        ":framework_internal",
        ":graph",
        ":lib",
        ":lib_experimental",
        ":lib_internal",
        ":protos_all_cc",
        "//tensorflow/core/debug:debug_graph_utils",
        "//tensorflow/core/kernels:function_ops",
        "//tensorflow/core/profiler/lib:profiler_lib",
        "//tensorflow/core/profiler/lib:profiler_session",
        "//tensorflow/core/profiler/lib:traceme",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
    alwayslink = 1,
)

cc_library(
    name = "example_parser_configuration",
    srcs = ["example/example_parser_configuration.cc"],
    hdrs = ["example/example_parser_configuration.h"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":core_cpu",
        ":core_cpu_internal",
        ":framework",
        ":lib",
        ":lib_internal",
        ":protos_all_cc",
    ],
    alwayslink = 1,
)

tf_proto_library_cc(
    name = "replay_log_proto",
    srcs = ["protobuf/replay_log.proto"],
    cc_api_version = 2,
    protodeps = [
        ":master_proto",
    ] + tf_additional_all_protos(),
    visibility = [
        "//tensorflow:internal",
    ],
)

cc_library(
    name = "gpu_id",
    hdrs = [
        "common_runtime/gpu/gpu_id.h",
        "common_runtime/gpu/gpu_id_manager.h",
    ],
    deps = [
        ":lib",
    ] + if_static([
        ":gpu_id_impl",
    ]),
)

cc_library(
    name = "gpu_id_impl",
    srcs = ["common_runtime/gpu/gpu_id_manager.cc"],
    hdrs = [
        "common_runtime/gpu/gpu_id.h",
        "common_runtime/gpu/gpu_id_manager.h",
    ],
    deps = [
        ":lib",
    ],
)

GPU_RUNTIME_HEADERS = [
    "common_runtime/gpu/gpu_bfc_allocator.h",
    "common_runtime/gpu/gpu_cudamalloc_allocator.h",
    "common_runtime/gpu/gpu_debug_allocator.h",
    "common_runtime/gpu/gpu_device.h",
    "common_runtime/gpu/gpu_host_allocator.h",
    "common_runtime/gpu/gpu_id.h",
    "common_runtime/gpu/gpu_id_manager.h",
    "common_runtime/gpu/gpu_id_utils.h",
    "common_runtime/gpu/gpu_init.h",
    "common_runtime/gpu/gpu_managed_allocator.h",
    "common_runtime/gpu/gpu_mem_allocator.h",
    "common_runtime/gpu/gpu_process_state.h",
    "common_runtime/gpu/gpu_stream_util.h",
    "common_runtime/gpu/gpu_util.h",
    "common_runtime/gpu_device_context.h",
]

tf_cuda_library(
    name = "gpu_runtime_impl",
    srcs = [
        "common_runtime/gpu/gpu_cudamalloc_allocator.cc",
        "common_runtime/gpu/gpu_debug_allocator.cc",
        "common_runtime/gpu/gpu_device.cc",
        "common_runtime/gpu/gpu_device_factory.cc",
        "common_runtime/gpu/gpu_managed_allocator.cc",
        "common_runtime/gpu/gpu_process_state.cc",
        "common_runtime/gpu/gpu_stream_util.cc",
        "common_runtime/gpu/gpu_util.cc",
        "common_runtime/gpu/gpu_util_platform_specific.cc",
    ],
    hdrs = GPU_RUNTIME_HEADERS,
    copts = tf_copts(),
    deps = [
        ":core_cpu_impl",
        ":core_cpu_lib",
        ":framework",
        ":framework_internal",
        ":gpu_bfc_allocator",
        ":gpu_id_impl",
        ":gpu_init_impl",
        ":gpu_lib",
        ":graph",
        ":lib",
        ":lib_internal",
        ":protos_all_cc",
        ":stream_executor",
        "//third_party/eigen3",
    ],
    alwayslink = 1,
)

tf_cuda_library(
    name = "gpu_runtime",
    hdrs = GPU_RUNTIME_HEADERS,
    linkstatic = 1,
    deps = [
        ":core_cpu_lib",
        ":framework",
        ":framework_internal",
        ":lib",
        ":lib_internal",
        ":protos_all_cc",
        ":stream_executor",
        "//third_party/eigen3",
    ] + if_static([":gpu_runtime_impl"]),
)

# This is redundant with the "gpu_runtime_*" targets above. It's useful for
# applications that want to depend on a minimal subset of TensorFlow (e.g. XLA).
tf_cuda_library(
    name = "gpu_bfc_allocator",
    srcs = [
        "common_runtime/gpu/gpu_bfc_allocator.cc",
    ],
    hdrs = ["common_runtime/gpu/gpu_bfc_allocator.h"],
    features = ["parse_headers"],
    visibility = ["//visibility:public"],
    deps = [
        ":bfc_allocator",
        ":gpu_mem_allocator",
        ":lib",
        ":lib_internal",
        ":protos_all_cc",
    ],
)

tf_cuda_library(
    name = "gpu_mem_allocator",
    srcs = [
        "common_runtime/gpu/gpu_id.h",
    ],
    hdrs = [
        "common_runtime/gpu/gpu_host_allocator.h",
        "common_runtime/gpu/gpu_mem_allocator.h",
    ],
    features = ["parse_headers"],
    visibility = ["//visibility:public"],
    deps = [
        ":allocator",
        ":lib",
        ":lib_internal",
        ":stream_executor",
    ],
)

tf_cuda_library(
    name = "gpu_init",
    hdrs = [
        "common_runtime/gpu/gpu_init.h",
    ],
    deps = [
        ":framework",
        ":framework_internal",
        ":lib",
        ":lib_internal",
        ":stream_executor",
    ] + if_static(
        [":gpu_init_impl"],
    ),
)

tf_cuda_library(
    name = "gpu_init_impl",
    srcs = [
        "common_runtime/gpu/gpu_init.cc",
    ],
    hdrs = [
        "common_runtime/gpu/gpu_init.h",
    ],
    copts = tf_copts(),
    linkstatic = 1,
    deps = [
        ":framework",
        ":framework_internal",
        ":lib",
        ":lib_internal",
        ":stream_executor",
    ],
    alwayslink = 1,
)

cc_library(
    name = "sycl_runtime",
    srcs = if_not_windows([
        "common_runtime/sycl/sycl_allocator.cc",
        "common_runtime/sycl/sycl_device.cc",
        "common_runtime/sycl/sycl_device_context.cc",
        "common_runtime/sycl/sycl_device_factory.cc",
    ]),
    hdrs = if_not_windows([
        "common_runtime/sycl/sycl_allocator.h",
        "common_runtime/sycl/sycl_device.h",
        "common_runtime/sycl/sycl_util.h",
        "common_runtime/sycl/sycl_device_context.h",
    ]),
    copts = tf_copts(),
    linkstatic = 0,
    deps = [
        ":core_cpu",
        ":core_cpu_internal",
        ":framework",
        ":framework_internal",
        ":lib",
        ":lib_internal",
        "//third_party/eigen3",
        "@local_config_sycl//sycl",
    ],
    alwayslink = 0,
)

# -----------------------------------------------------------------------------
# Tests

cc_library(
    name = "lib_test_internal",
    testonly = 1,
    hdrs = [
        "//tensorflow/core/lib/gtl:legacy_lib_test_internal_headers",
        "//tensorflow/core/lib/io:legacy_lib_test_internal_headers",
        "//tensorflow/core/lib/random:legacy_lib_test_internal_headers",
    ],
    deps = [
        ":lib",
        ":lib_internal",
    ],
)

cc_library(
    name = "tensor_testutil",
    testonly = 1,
    srcs = ["framework/tensor_testutil.cc"],
    hdrs = ["framework/tensor_testutil.h"],
    copts = tf_copts(),
    deps = [
        ":framework",
        ":lib",
        ":test",
    ],
)

cc_library(
    name = "shape_inference_testutil",
    testonly = 1,
    srcs = ["framework/shape_inference_testutil.cc"],
    hdrs = ["framework/shape_inference_testutil.h"],
    copts = tf_copts(),
    deps = [
        ":framework",
        ":lib",
        ":lib_internal",
        ":protos_all_cc",
    ],
)

# Main program for tests
cc_library(
    name = "test_main",
    testonly = 1,
    srcs = ["//tensorflow/core/platform:test_main.cc"],
    copts = tf_copts(),
    linkopts = select({
        "//tensorflow:windows": [],
        "//conditions:default": ["-lm"],
    }),
    visibility = ["//tensorflow:internal"],
    deps = [
        ":lib",
        ":lib_internal",
        ":test",  # buildcleaner: keep
        "//tensorflow/core/platform/default/build_config:test_main",
    ],
    alwayslink = 1,
)

tf_cc_tests(
    name = "low_level_library_tests",
    size = "small",
    srcs = [
        "lib/histogram/histogram_test.cc",
        "lib/monitoring/collection_registry_test.cc",
        "lib/monitoring/counter_test.cc",
        "lib/monitoring/gauge_test.cc",
        "lib/monitoring/metric_def_test.cc",
        "lib/monitoring/sampler_test.cc",
        "lib/wav/wav_io_test.cc",
        "//tensorflow/core/lib/core:legacy_lib_core_all_tests",
        "//tensorflow/core/lib/gtl:legacy_lib_gtl_tests",
        "//tensorflow/core/lib/hash:legacy_lib_hash_all_tests",
        "//tensorflow/core/lib/io:legacy_lib_io_all_tests",
        "//tensorflow/core/lib/math:math_util_test.cc",
        "//tensorflow/core/lib/random:legacy_lib_random_tests",
        "//tensorflow/core/lib/strings:legacy_low_level_library_tests",
        "//tensorflow/core/platform:fingerprint_test.cc",
        "//tensorflow/core/platform:integral_types_test.cc",
        "//tensorflow/core/platform:logging_test.cc",
        "//tensorflow/core/platform:mutex_test.cc",
        "//tensorflow/core/platform:net_test.cc",
        "//tensorflow/core/platform:port_test.cc",
        "//tensorflow/core/platform:profile_utils/cpu_utils_test.cc",
        "//tensorflow/core/platform:scanner_test.cc",
        "//tensorflow/core/platform:stacktrace_handler_test.cc",
        "//tensorflow/core/platform:stacktrace_test.cc",
        "//tensorflow/core/platform:str_util_test.cc",
        "//tensorflow/core/platform:strcat_test.cc",
        "//tensorflow/core/platform:stringpiece_test.cc",
        "//tensorflow/core/platform:stringprintf_test.cc",
        "//tensorflow/core/platform:subprocess_test.cc",
        "//tensorflow/core/platform:vmodule_benchmark_test.cc",
    ],
    deps = [
        ":core_cpu_internal",
        ":lib",
        ":lib_internal",
        ":lib_test_internal",
        ":protos_all_cc",
        ":test",
        ":test_main",
        "//tensorflow/core/platform:scanner",
        "//tensorflow/core/platform:str_util",
        "//tensorflow/core/platform:strcat",
        "//tensorflow/core/platform:stringpiece",
        "//tensorflow/core/platform:stringprintf",
        "//third_party/eigen3",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:optional",
        "@zlib_archive//:zlib",
    ],
)

tf_cc_test(
    name = "vmodule_test",
    srcs = ["//tensorflow/core/platform:vmodule_test.cc"],
    tags = ["optonly"],
    deps = [
        ":lib",
        ":lib_internal",
        ":lib_test_internal",
        ":protos_all_cc",
        ":test",
        "//third_party/eigen3",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "lib_random_random_distributions_test",
    srcs = ["//tensorflow/core/lib/random:legacy_lib_random_random_distributions_test"],
    tags = ["optonly"],
    deps = [
        ":lib",
        ":lib_internal",
        ":lib_test_internal",
        ":protos_all_cc",
        ":test",
        ":test_main",
        "//third_party/eigen3",
    ],
)

tf_cc_test(
    name = "platform_strings_test",
    size = "small",
    srcs = ["//tensorflow/core/platform:platform_strings_test.cc"],
    features = ["-dynamic_link_test_srcs"],  # see go/dynamic_link_test_srcs
    deps = [
        ":lib",
        "//tensorflow/core/platform:platform_strings",
    ],
)

tf_cc_test(
    name = "platform_env_test",
    size = "small",
    srcs = ["//tensorflow/core/platform:env_test.cc"],
    deps = [
        ":lib",
        ":lib_internal",
        ":lib_test_internal",
        ":protos_all_cc",
        ":test",
        ":test_main",
        "//third_party/eigen3",
    ],
)

tf_cc_test(
    name = "platform_fake_python_env_test",
    size = "small",
    srcs = ["//tensorflow/core/platform:fake_python_env_test.cc"],
    args = [
        "/some/path/to/pythontest.runfiles/org_tensorflow/stuff/to/run.py",
    ],
    tags = [
        "local",
        "no_windows",
        "nogpu",
        "nomac",
        "notap",
    ],
    deps = [
        ":lib",
        ":lib_internal",
        ":lib_test_internal",
        ":test",
        ":test_main",
    ],
)

tf_cc_test(
    name = "platform_abi_test",
    size = "small",
    srcs = ["//tensorflow/core/platform:abi_test.cc"],
    deps = [
        ":framework",
        ":lib",
        ":lib_internal",
        ":lib_test_internal",
        ":protos_all_cc",
        ":test",
        ":test_main",
        "//third_party/eigen3",
    ],
)

tf_cc_test(
    name = "platform_numa_test",
    size = "small",
    srcs = ["//tensorflow/core/platform:numa_test.cc"],
    tags = [
        # This test will not pass unless it has access to all NUMA nodes
        # on the executing machine.
        "manual",
        "notap",
    ],
    deps = [
        ":framework",
        ":lib",
        ":lib_internal",
        ":lib_test_internal",
        ":protos_all_cc",
        ":test",
        ":test_main",
        "//third_party/eigen3",
    ],
)

tf_cc_test(
    name = "platform_setround_test",
    size = "small",
    srcs = ["//tensorflow/core/platform:setround_test.cc"],
    tags = [
        "noasan",
        "noclang",
        "nomsan",
        "notsan",
    ],
    deps = [
        ":lib",
        ":lib_internal",
        ":lib_test_internal",
        ":test",
        ":test_main",
    ],
)

tf_cc_test(
    name = "platform_file_system_test",
    size = "small",
    srcs = ["//tensorflow/core/platform:file_system_test.cc"],
    deps = [
        ":lib",
        ":lib_internal",
        ":lib_test_internal",
        ":protos_all_cc",
        ":test",
        ":test_main",
    ],
)

tf_cc_test(
    name = "util_overflow_test",
    size = "small",
    srcs = ["util/overflow_test.cc"],
    deps = [
        ":framework_lite",
        ":overflow",
        ":test",
        ":test_main",
    ],
)

tf_cc_test(
    name = "exec_on_stall_test",
    size = "small",
    srcs = ["util/exec_on_stall_test.cc"],
    deps = [
        ":exec_on_stall",
        ":framework_lite",
        ":test",
        ":test_main",
    ],
)

tf_cc_test(
    name = "lib_jpeg_jpeg_mem_unittest",
    srcs = ["lib/jpeg/jpeg_mem_unittest.cc"],
    data = glob(["lib/jpeg/testdata/*.jpg"]),
    deps = [
        ":jpeg_internal",
        ":lib",
        ":lib_internal",
        ":test",
        ":test_main",
        "@com_google_absl//absl/base",
    ],
)

tf_cc_test(
    name = "lib_strings_ordered_code_test",
    srcs = ["//tensorflow/core/lib/strings:legacy_strings_ordered_code_test"],
    extra_copts = ["$(STACK_FRAME_UNLIMITED)"],  # Tests initialize large vectors
    deps = [
        ":lib",
        ":lib_internal",
        ":test",
        ":test_main",
    ],
)

tf_cc_test(
    name = "lib_strings_proto_serialization_test",
    srcs = ["//tensorflow/core/lib/strings:legacy_strings_proto_serialization_test"],
    deps = [
        ":lib",
        ":lib_internal",
        ":lib_test_internal",
        ":protos_all_cc",
        ":test",
        ":test_main",
        "@com_google_absl//absl/memory",
    ],
)

tf_cc_test(
    name = "lib_random_weighted_picker_test",
    size = "medium",
    srcs = ["//tensorflow/core/lib/random:legacy_lib_random_random_weighted_picker_test"],
    deps = [
        ":lib",
        ":lib_internal",
        ":test",
        ":test_main",
    ],
)

tf_cc_test(
    name = "framework_op_gen_lib_test",
    size = "small",
    srcs = ["framework/op_gen_lib_test.cc"],
    deps = [
        ":op_gen_lib",
        ":protos_all_cc",
        ":test",
        ":test_main",
    ],
)

tf_cc_test(
    name = "quantize_training_test",
    srcs = ["graph/quantize_training_test.cc"],
    deps = [
        ":all_kernels",
        ":core",
        ":core_cpu",
        ":core_cpu_internal",
        ":direct_session_internal",
        ":framework",
        ":framework_internal",
        ":lib",
        ":lib_internal",
        ":ops",
        ":protos_all_cc",
        ":protos_test_cc",
        ":test",
        ":test_main",
        ":testlib",
    ],
)

tf_cc_tests(
    name = "higher_level_tests",
    size = "small",
    srcs = [
        "common_runtime/buf_rendezvous_test.cc",
        "common_runtime/collective_executor_mgr_test.cc",
        "common_runtime/collective_rma_local_test.cc",
        "common_runtime/device_resolver_local_test.cc",
        "common_runtime/device_set_test.cc",
        "common_runtime/dynamic_device_mgr_test.cc",
        "common_runtime/isolate_placer_inspection_required_ops_pass_test.cc",
        "common_runtime/optimization_registry_test.cc",
        "common_runtime/pending_counts_test.cc",
        "common_runtime/placer_inspection_required_ops_utils_test.cc",
        "common_runtime/placer_test.cc",
        "common_runtime/session_test.cc",
        "common_runtime/threadpool_device_test.cc",
        "example/feature_util_test.cc",
        "framework/allocator_test.cc",
        "framework/attr_value_util_test.cc",
        "framework/bfloat16_test.cc",
        "framework/cancellation_test.cc",
        "framework/common_shape_fns_test.cc",
        "framework/dataset_test.cc",
        "framework/device_base_test.cc",
        "framework/function_test.cc",
        "framework/graph_def_util_test.cc",
        "framework/graph_to_functiondef_test.cc",
        "framework/kernel_def_builder_test.cc",
        "framework/kernel_def_util_test.cc",
        "framework/memory_types_test.cc",
        "framework/model_test.cc",
        "framework/node_def_builder_test.cc",
        "framework/node_def_util_test.cc",
        "framework/op_compatibility_test.cc",
        "framework/op_def_builder_test.cc",
        "framework/op_def_util_test.cc",
        "framework/op_kernel_test.cc",
        "framework/op_registration_test.cc",
        "framework/partial_tensor_shape_test.cc",
        "framework/rendezvous_test.cc",
        "framework/resource_mgr_test.cc",
        "framework/resource_op_kernel_test.cc",
        "framework/shape_inference_test.cc",
        "framework/shape_inference_testutil_test.cc",
        "framework/tensor_shape_test.cc",
        "framework/tensor_slice_test.cc",
        "framework/tensor_test.cc",
        "framework/tensor_testutil_test.cc",
        "framework/tensor_util_test.cc",
        "framework/tracking_allocator_test.cc",
        "framework/types_test.cc",
        "framework/unique_tensor_references_test.cc",
        "framework/variant_op_registry_test.cc",
        "framework/variant_test.cc",
        "graph/algorithm_test.cc",
        "graph/control_flow_test.cc",
        "graph/edgeset_test.cc",
        "graph/graph_def_builder_test.cc",
        "graph/graph_partition_test.cc",
        "graph/graph_test.cc",
        "graph/node_builder_test.cc",
        "graph/optimizer_cse_test.cc",
        "graph/subgraph_test.cc",
        "graph/tensor_id_test.cc",
        "graph/validate_test.cc",
        "util/bcast_test.cc",
        "util/command_line_flags_test.cc",
        "util/debug_events_writer_test.cc",
        "util/device_name_utils_test.cc",
        "util/dump_graph_test.cc",
        "util/equal_graph_def_test.cc",
        "util/events_writer_test.cc",
        "util/example_proto_fast_parsing_test.cc",
        "util/example_proto_helper_test.cc",
        "util/matmul_bcast_test.cc",
        "util/memmapped_file_system_test.cc",
        "util/presized_cuckoo_map_test.cc",
        "util/reffed_status_callback_test.cc",
        "util/reporter_test.cc",
        "util/saved_tensor_slice_util_test.cc",
        "util/semver_test.cc",
        "util/sparse/sparse_tensor_test.cc",
        "util/stat_summarizer_test.cc",
        "util/tensor_format_test.cc",
        "util/tensor_slice_reader_test.cc",
        "util/tensor_slice_set_test.cc",
        "util/tensor_slice_util_test.cc",
        "util/tensor_slice_writer_test.cc",
        "util/work_sharder_test.cc",
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
        ":framework",
        ":framework_internal",
        ":lib",
        ":lib_internal",
        ":ops",
        ":protos_all_cc",
        ":protos_test_cc",
        ":test",
        ":test_main",
        ":testlib",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:cc_ops_internal",
        "//tensorflow/cc:function_ops",
        "//tensorflow/cc:ops",
        "//tensorflow/cc:scope",
        "//tensorflow/cc:sendrecv_ops",
        "//tensorflow/cc:while_loop",
        "//tensorflow/core/kernels:ops_util",
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
        "common_runtime/collective_param_resolver_local_test.cc",
        "graph/graph_constructor_test.cc",
    ],
    linkopts = select({
        "//tensorflow:macos": ["-headerpad_max_install_names"],
        "//conditions:default": [],
    }),
    linkstatic = tf_kernel_tests_linkstatic(),
    deps = [
        ":all_kernels",
        ":core",
        ":core_cpu",
        ":core_cpu_internal",
        ":direct_session_internal",
        ":framework",
        ":framework_internal",
        ":lib",
        ":lib_internal",
        ":ops",
        ":protos_all_cc",
        ":protos_test_cc",
        ":test",
        ":test_main",
        ":testlib",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:cc_ops_internal",
        "//tensorflow/cc:scope",
        "//tensorflow/cc:sendrecv_ops",
        "//tensorflow/core/kernels:ops_util",
        "//third_party/eigen3",
    ],
)

tf_cc_test(
    name = "cudnn_rnn_ops_test_cc",
    size = "small",
    srcs = [
        "ops/cudnn_rnn_ops_test.cc",
    ],
    deps = [
        ":core",
        ":framework",
        ":lib",
        ":test",
        ":test_main",
        ":testlib",
    ],
)

tf_cc_tests(
    name = "collective_order_test",
    size = "small",
    srcs = [
        "graph/collective_order_test.cc",
    ],
    deps = [
        ":core",
        ":core_cpu",
        ":core_cpu_internal",
        ":framework",
        ":framework_internal",
        ":lib",
        ":lib_internal",
        ":ops",
        ":protos_all_cc",
        ":test",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_cc_tests_gpu(
    name = "ring_reducer_test",
    size = "medium",
    srcs = [
        "common_runtime/ring_reducer_test.cc",
    ],
    linkstatic = tf_kernel_tests_linkstatic(),
    tags = ["no_cuda_on_cpu_tap"],
    deps = [
        ":all_kernels",
        ":core",
        ":core_cpu",
        ":core_cpu_internal",
        ":direct_session_internal",
        ":framework",
        ":framework_internal",
        ":gpu_runtime",
        ":lib",
        ":lib_internal",
        ":ops",
        ":protos_all_cc",
        ":protos_test_cc",
        ":test",
        ":test_main",
        ":testlib",
        "@com_google_absl//absl/memory",
    ],
)

tf_cc_tests_gpu(
    name = "ring_gatherer_test",
    size = "medium",
    srcs = [
        "common_runtime/ring_gatherer_test.cc",
    ],
    linkstatic = tf_kernel_tests_linkstatic(),
    tags = ["no_cuda_on_cpu_tap"],
    deps = [
        ":all_kernels",
        ":core",
        ":core_cpu",
        ":core_cpu_internal",
        ":direct_session_internal",
        ":framework",
        ":framework_internal",
        ":gpu_runtime",
        ":lib",
        ":lib_internal",
        ":ops",
        ":protos_all_cc",
        ":protos_test_cc",
        ":test",
        ":test_main",
        ":testlib",
        "@com_google_absl//absl/memory",
    ],
)

tf_cc_tests_gpu(
    name = "hierarchical_tree_broadcaster_test",
    size = "medium",
    srcs = [
        "common_runtime/hierarchical_tree_broadcaster_test.cc",
    ],
    linkstatic = tf_kernel_tests_linkstatic(),
    tags = ["no_cuda_on_cpu_tap"],
    deps = [
        ":all_kernels",
        ":core",
        ":core_cpu",
        ":core_cpu_internal",
        ":direct_session_internal",
        ":framework",
        ":framework_internal",
        ":gpu_runtime",
        ":lib",
        ":lib_internal",
        ":ops",
        ":protos_all_cc",
        ":protos_test_cc",
        ":test",
        ":test_main",
        ":testlib",
        "@com_google_absl//absl/memory",
    ],
)

tf_cc_test_mkl(
    name = "mkl_runtime_tests",
    size = "small",
    srcs = [
        "common_runtime/mkl_cpu_allocator_test.cc",
        "common_runtime/mkl_threadpool_device_test.cc",
    ],
    linkstatic = 1,
    deps = [
        ":core",
        ":core_cpu",
        ":core_cpu_internal",
        ":framework",
        ":framework_internal",
        ":lib",
        ":test",
        ":test_main",
        ":testlib",
    ],
)

tf_cc_test_mkl(
    name = "mkl_related_tests",
    size = "small",
    srcs = [
        "graph/mkl_layout_pass_test.cc",
        "graph/mkl_tfconversion_pass_test.cc",
        "util/mkl_util_test.cc",
    ],
    linkstatic = 1,
    deps = [
        ":core",
        ":core_cpu",
        ":core_cpu_internal",
        ":direct_session_internal",
        ":framework",
        ":framework_internal",
        ":lib",
        ":lib_internal",
        ":ops",
        ":protos_all_cc",
        ":test",
        ":test_main",
        ":testlib",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:scope",
        "//tensorflow/cc:sendrecv_ops",
        "//tensorflow/core/kernels:ops_util",
        "//third_party/eigen3",
    ] + if_mkl([
        "//tensorflow/core/kernels:mkl_aggregate_ops",
        "//tensorflow/core/kernels:mkl_batch_matmul_op",
        "//tensorflow/core/kernels:mkl_concat_op",
        "//tensorflow/core/kernels:mkl_conv_op",
        "//tensorflow/core/kernels:mkl_cwise_ops_common",
        "//tensorflow/core/kernels:mkl_dequantize_op",
        "//tensorflow/core/kernels:mkl_fused_batch_norm_op",
        "//tensorflow/core/kernels:mkl_identity_op",
        "//tensorflow/core/kernels:mkl_input_conversion_op",
        "//tensorflow/core/kernels:mkl_lrn_op",
        "//tensorflow/core/kernels:mkl_matmul_op",
        "//tensorflow/core/kernels:mkl_pooling_ops",
        "//tensorflow/core/kernels:mkl_qmatmul_op",
        "//tensorflow/core/kernels:mkl_quantize_op",
        "//tensorflow/core/kernels:mkl_relu_op",
        "//tensorflow/core/kernels:mkl_reshape_op",
        "//tensorflow/core/kernels:mkl_slice_op",
        "//tensorflow/core/kernels:mkl_softmax_op",
        "//tensorflow/core/kernels:mkl_tfconv_op",
        "//tensorflow/core/kernels:mkl_transpose_op",
    ]),
)

tf_cc_tests_gpu(
    name = "gpu_device_on_non_gpu_machine_test",
    size = "small",
    srcs = ["common_runtime/gpu/gpu_device_on_non_gpu_machine_test.cc"],
    linkstatic = tf_kernel_tests_linkstatic(),
    deps = [
        ":gpu_headers_lib",
        ":gpu_id",
        ":gpu_runtime",
        ":test",
    ],
)

tf_cc_tests_gpu(
    name = "gpu_related_tests",
    size = "small",
    srcs = glob(["user_ops/**/*_test.cc"]) + [
        "common_runtime/gpu/gpu_bfc_allocator_test.cc",
        "common_runtime/gpu/gpu_device_test.cc",
        "common_runtime/gpu/gpu_id_manager_test.cc",
        "common_runtime/gpu/pool_allocator_test.cc",
    ],
    linkstatic = tf_kernel_tests_linkstatic(),
    tags = tf_cuda_tests_tags(),
    deps = [
        ":core_cpu",
        ":core_cpu_internal",
        ":direct_session",
        ":framework",
        ":framework_internal",
        ":gpu_id",
        ":gpu_runtime",
        ":lib",
        ":lib_internal",
        ":protos_all_cc",
        ":test",
        ":test_main",
        ":testlib",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core/kernels:ops_util",
    ],
)

tf_cc_test_gpu(
    name = "gpu_event_mgr_test",
    srcs = ["common_runtime/gpu/gpu_event_mgr_test.cc"],
    linkstatic = tf_kernel_tests_linkstatic(),
    tags = tf_cuda_tests_tags(),
    deps = [
        ":framework",
        ":framework_internal",
        ":lib",
        ":lib_internal",
        ":protos_all_cc",
        ":test",
        ":test_main",
        ":testlib",
        "//tensorflow/core/kernels:cwise_op",
    ],
)

tf_cuda_cc_test(
    name = "gpu_device_unified_memory_test",
    size = "small",
    srcs = [
        "common_runtime/gpu/gpu_device_test.cc",
    ],
    linkstatic = tf_kernel_tests_linkstatic(),
    # Runs test on a Guitar cluster that uses P100s to test unified memory
    # allocations.
    tags = tf_cuda_tests_tags() + [
        "guitar",
        "multi_gpu",
    ],
    deps = [
        ":core_cpu",
        ":core_cpu_internal",
        ":direct_session",
        ":framework",
        ":framework_internal",
        ":gpu_id",
        ":lib",
        ":lib_internal",
        ":protos_all_cc",
        ":test",
        ":test_main",
        ":testlib",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core/kernels:ops_util",
    ],
)

tf_cc_test_gpu(
    name = "rocm_rocdl_path_test",
    size = "small",
    srcs = ["//tensorflow/core/platform:rocm_rocdl_path_test.cc"],
    linkstatic = tf_kernel_tests_linkstatic(),
    tags = tf_gpu_tests_tags(),
    deps = [
        ":lib",
        ":test",
        ":test_main",
        "//tensorflow/core/platform:rocm_rocdl_path",
    ],
)

tf_cuda_only_cc_test(
    name = "util_gpu_kernel_helper_test",
    srcs = [
        "util/gpu_kernel_helper_test.cu.cc",
    ],
    deps = [
        ":test",
        ":test_main",
        "//third_party/eigen3",
    ] + mkl_deps(),
)

tf_cc_test_gpu(
    name = "memory_types_test",
    size = "small",
    srcs = ["common_runtime/memory_types_test.cc"],
    linkstatic = tf_kernel_tests_linkstatic(),
    tags = tf_cuda_tests_tags(),
    deps = [
        ":core",
        ":core_cpu",
        ":core_cpu_internal",
        ":framework",
        ":framework_internal",
        ":gpu_runtime",
        ":lib",
        ":lib_internal",
        ":ops",
        ":protos_all_cc",
        ":test",
        ":test_main",
        ":testlib",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core/kernels:cast_op",
        "//third_party/eigen3",
    ],
)

tf_cc_test_gpu(
    name = "variant_op_copy_test",
    size = "small",
    srcs = ["framework/variant_op_copy_test.cc"],
    linkstatic = tf_kernel_tests_linkstatic(),
    tags = tf_cuda_tests_tags(),
    deps = [
        ":core",
        ":core_cpu",
        ":core_cpu_internal",
        ":direct_session",
        ":framework",
        ":framework_internal",
        ":gpu_runtime",
        ":lib",
        ":lib_internal",
        ":protos_all_cc",
        ":test",
        ":test_main",
        ":testlib",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:client_session",
        "//tensorflow/cc:ops",
        "//tensorflow/cc:scope",
        "//tensorflow/core/kernels:array",
        "//third_party/eigen3",
    ],
)

tf_cc_test(
    name = "common_runtime_constant_folding_test",
    size = "small",
    srcs = ["common_runtime/constant_folding_test.cc"],
    linkstatic = tf_kernel_tests_linkstatic(),
    tags = tf_cuda_tests_tags() + ["no_rocm"],
    deps = [
        ":core",
        ":core_cpu",
        ":core_cpu_internal",
        ":direct_session_internal",
        ":framework",
        ":framework_internal",
        ":gpu_runtime",
        ":lib",
        ":lib_internal",
        ":ops",
        ":protos_all_cc",
        ":test",
        ":test_main",
        ":testlib",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:cc_ops_internal",
        "//tensorflow/cc:sendrecv_ops",
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
    name = "common_runtime_shape_refiner_test",
    size = "small",
    srcs = [
        "common_runtime/shape_refiner_test.cc",
    ],
    linkstatic = tf_kernel_tests_linkstatic(),
    deps = [
        ":core",
        ":core_cpu",
        ":core_cpu_internal",
        ":framework",
        ":framework_internal",
        ":lib",
        ":lib_internal",
        ":ops",
        ":protos_all_cc",
        ":test",
        ":test_main",
        ":testlib",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:resource_variable_ops",
        "//tensorflow/cc:scope",
        "//tensorflow/core/kernels:array",
        "//tensorflow/core/kernels:math",
        "//tensorflow/core/kernels:resource_variable_ops",
        "//third_party/eigen3",
    ],
)

tf_cuda_cc_test(
    name = "common_runtime_process_function_library_runtime_test",
    size = "small",
    srcs = ["common_runtime/process_function_library_runtime_test.cc"],
    linkstatic = tf_kernel_tests_linkstatic(),
    tags = ["no_rocm"],
    deps = [
        ":core_cpu",
        ":core_cpu_internal",
        ":framework",
        ":framework_internal",
        ":lib",
        ":protos_all_cc",
        ":test",
        ":test_main",
        ":testlib",
        "//tensorflow/cc:function_ops",
        "//tensorflow/core/kernels:cast_op",
        "//tensorflow/core/kernels:cwise_op",
        "//tensorflow/core/kernels:function_ops",
        "//tensorflow/core/kernels:resource_variable_ops",
    ],
)

tf_cc_test(
    name = "common_runtime_process_util_test",
    size = "small",
    srcs = ["common_runtime/process_util_test.cc"],
    linkstatic = tf_kernel_tests_linkstatic(),
    deps = [
        ":core_cpu_internal",
        ":test",
        ":test_main",
    ],
)

tf_cc_test(
    name = "common_runtime_rendezvous_util_test",
    size = "small",
    srcs = ["common_runtime/rendezvous_util_test.cc"],
    linkstatic = tf_kernel_tests_linkstatic(),
    deps = [
        ":core_cpu_internal",
        ":lib",
        ":test",
        ":test_main",
    ],
)

tf_cc_test(
    name = "framework_run_handler_util_test",
    size = "small",
    srcs = ["framework/run_handler_util_test.cc"],
    linkstatic = tf_kernel_tests_linkstatic(),
    deps = [
        ":framework_internal",
        ":lib",
        ":test",
        ":test_main",
    ],
)

tf_cc_test(
    name = "framework_run_handler_test",
    size = "small",
    srcs = ["framework/run_handler_test.cc"],
    linkstatic = tf_kernel_tests_linkstatic(),
    deps = [
        ":framework_internal",
        ":lib",
        ":lib_internal",
        ":test",
        ":test_main",
        "//third_party/eigen3",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/synchronization",
    ],
)

tf_cc_test(
    name = "common_runtime_partitioning_utils_test",
    size = "small",
    srcs = ["common_runtime/partitioning_utils_test.cc"],
    deps = [
        ":core_cpu",
        ":core_cpu_internal",
        ":framework",
        ":lib",
        ":ops",
        ":test",
        ":test_main",
        ":testlib",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:cc_ops_internal",
        "//tensorflow/cc:function_ops",
        "//tensorflow/core/kernels:function_ops",
        "//tensorflow/core/kernels:identity_op",
    ],
)

tf_cuda_cc_test(
    name = "common_runtime_direct_session_test",
    size = "small",
    srcs = ["common_runtime/direct_session_test.cc"],
    args = [] + if_cuda(["--heap_check=local"]),  # The GPU tracer leaks memory
    linkstatic = tf_kernel_tests_linkstatic(),
    deps = [
        ":core_cpu",
        ":core_cpu_internal",
        ":direct_session_internal",
        ":framework",
        ":framework_internal",
        ":lib",
        ":lib_internal",
        ":ops",
        ":protos_all_cc",
        ":test",
        ":test_main",
        ":testlib",
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
    ] + if_cuda([":cuda"]),
)

# This is identical to :common_runtime_direct_session_test with the addition of
# a dependency on alwayslink target //third_party/tensorflow/core/debug, which
# enables support for TensorFlow Debugger (tfdbg).
tf_cc_test(
    name = "common_runtime_direct_session_with_debug_test",
    size = "small",
    srcs = ["common_runtime/direct_session_test.cc"],
    linkstatic = tf_kernel_tests_linkstatic(),
    deps = [
        ":core",
        ":core_cpu",
        ":core_cpu_internal",
        ":direct_session_internal",
        ":framework",
        ":framework_internal",
        ":lib",
        ":lib_internal",
        ":ops",
        ":protos_all_cc",
        ":test",
        ":test_main",
        ":testlib",
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
    name = "common_runtime_direct_session_with_tracking_alloc_test",
    size = "small",
    srcs = ["common_runtime/direct_session_with_tracking_alloc_test.cc"],
    args = ["--heap_check=local"],  # The GPU tracer leaks memory
    linkstatic = tf_kernel_tests_linkstatic(),
    tags = ["no_gpu"],
    deps = [
        ":core",
        ":core_cpu",
        ":core_cpu_internal",
        ":direct_session_internal",
        ":framework",
        ":framework_internal",
        ":lib",
        ":lib_internal",
        ":ops",
        ":protos_all_cc",
        ":test",
        ":test_main",
        ":testlib",
        "//tensorflow/cc:cc_ops",
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
    name = "common_runtime_graph_runner_test",
    size = "small",
    srcs = ["common_runtime/graph_runner_test.cc"],
    linkstatic = tf_kernel_tests_linkstatic(),
    deps = [
        ":array_ops_op_lib",
        ":core",
        ":core_cpu",
        ":core_cpu_internal",
        ":direct_session_internal",
        ":framework",
        ":framework_internal",
        ":lib",
        ":lib_internal",
        ":ops",
        ":protos_all_cc",
        ":test",
        ":test_main",
        ":testlib",
        "//tensorflow/c/kernels:bitcast_op_lib",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:scope",
        "//tensorflow/core/kernels:cwise_op",
        "//third_party/eigen3",
    ] + if_mkl([":mkl_array_ops_op_lib"]),
)

tf_cc_test(
    name = "common_runtime_executor_test",
    size = "small",
    srcs = ["common_runtime/executor_test.cc"],
    linkstatic = tf_kernel_tests_linkstatic(),
    deps = [
        ":core",
        ":core_cpu",
        ":core_cpu_internal",
        ":framework",
        ":framework_internal",
        ":lib",
        ":lib_internal",
        ":protos_all_cc",
        ":test",
        ":test_main",
        ":testlib",
        "//tensorflow/core/kernels:array",
        "//tensorflow/core/kernels:control_flow_ops",
        "//tensorflow/core/kernels:math",
        "//tensorflow/core/kernels:random_ops",
        "//tensorflow/core/kernels:state",
    ],
)

tf_cc_test(
    name = "common_runtime_function_test",
    size = "small",
    srcs = ["common_runtime/function_test.cc"],
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
        ":framework",
        ":framework_internal",
        ":lib",
        ":lib_internal",
        ":ops",
        ":protos_all_cc",
        ":test",
        ":test_main",
        ":testlib",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:cc_ops_internal",
        "//tensorflow/cc:function_ops",
        "//tensorflow/cc:functional_ops",
        "//tensorflow/cc:sendrecv_ops",
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
    name = "common_runtime_function_threadpool_test",
    size = "small",
    srcs = ["common_runtime/function_threadpool_test.cc"],
    linkstatic = tf_kernel_tests_linkstatic(),
    deps = [
        ":core",
        ":core_cpu",
        ":core_cpu_internal",
        ":direct_session_internal",
        ":framework",
        ":framework_internal",
        ":lib",
        ":lib_internal",
        ":ops",
        ":protos_all_cc",
        ":test",
        ":test_main",
        ":testlib",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:cc_ops_internal",
        "//tensorflow/cc:function_ops",
        "//tensorflow/cc:functional_ops",
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
    name = "common_runtime_scoped_allocator_mgr_test",
    size = "small",
    srcs = ["common_runtime/scoped_allocator_mgr_test.cc"],
    linkstatic = tf_kernel_tests_linkstatic(),
    deps = [
        ":core_cpu",
        ":core_cpu_internal",
        ":framework",
        ":lib",
        ":test",
        ":test_main",
    ],
)

tf_cc_test_gpu(
    name = "gpu_allocator_retry_test",
    size = "medium",
    srcs = ["common_runtime/gpu/gpu_allocator_retry_test.cc"],
    linkstatic = tf_kernel_tests_linkstatic(),
    tags = tf_cuda_tests_tags(),
    deps = [
        ":core_cpu",
        ":core_cpu_internal",
        ":direct_session",
        ":framework",
        ":framework_internal",
        ":gpu_runtime",
        ":lib",
        ":lib_internal",
        ":protos_all_cc",
        ":test",
        ":test_main",
        ":testlib",
        "//tensorflow/cc:cc_ops",
    ],
)

tf_cc_test_gpu(
    name = "gpu_debug_allocator_test",
    size = "medium",
    srcs = ["common_runtime/gpu/gpu_debug_allocator_test.cc"],
    args = ["--gtest_death_test_style=threadsafe"],
    linkstatic = tf_kernel_tests_linkstatic(),
    tags = tf_cuda_tests_tags(),
    deps = [
        ":core_cpu",
        ":core_cpu_internal",
        ":direct_session",
        ":framework",
        ":framework_internal",
        ":gpu_id",
        ":gpu_runtime",
        ":lib",
        ":lib_internal",
        ":protos_all_cc",
        ":test",
        ":test_main",
        ":testlib",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core/kernels:ops_util",
    ],
)

tf_cc_test_gpu(
    name = "gpu_stream_util_test",
    size = "small",
    srcs = ["common_runtime/gpu/gpu_stream_util_test.cc"],
    linkstatic = tf_kernel_tests_linkstatic(),
    tags = tf_cuda_tests_tags() + ["nomac"],
    deps = [
        ":core_cpu",
        ":core_cpu_internal",
        ":direct_session",
        ":framework",
        ":framework_internal",
        ":gpu_runtime",
        ":lib",
        ":lib_internal",
        ":protos_all_cc",
        ":test",
        ":test_main",
        ":testlib",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:sendrecv_ops",
        "//tensorflow/core/kernels:matmul_op",
        "//tensorflow/core/kernels:ops_util",
    ],
)

tf_cc_test(
    name = "framework_op_segment_test",
    size = "small",
    srcs = ["framework/op_segment_test.cc"],
    linkstatic = tf_kernel_tests_linkstatic(),
    deps = [
        ":core",
        ":core_cpu",
        ":core_cpu_internal",
        ":direct_session_internal",
        ":framework",
        ":framework_internal",
        ":lib",
        ":lib_internal",
        ":ops",
        ":protos_all_cc",
        ":test",
        ":test_main",
        ":testlib",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core/kernels:cwise_op",
        "//tensorflow/core/kernels:ops_util",
        "//third_party/eigen3",
    ],
)

tf_cc_test(
    name = "ops_array_grad_test",
    size = "small",
    srcs = ["ops/array_grad_test.cc"],
    linkstatic = tf_kernel_tests_linkstatic(),
    deps = [
        ":core",
        ":core_cpu",
        ":core_cpu_internal",
        ":direct_session_internal",
        ":framework",
        ":framework_internal",
        ":lib",
        ":lib_internal",
        ":ops",
        ":protos_all_cc",
        ":test",
        ":test_main",
        ":testlib",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core/kernels:array",
        "//tensorflow/core/kernels:cwise_op",
        "//tensorflow/core/kernels:function_ops",
        "//tensorflow/core/kernels:math",
        "//third_party/eigen3",
    ],
)

tf_cc_test(
    name = "ops_math_grad_test",
    size = "small",
    srcs = ["ops/math_grad_test.cc"],
    linkstatic = tf_kernel_tests_linkstatic(),
    tags = ["no_gpu"],
    deps = [
        ":core",
        ":core_cpu",
        ":core_cpu_internal",
        ":direct_session_internal",
        ":framework",
        ":framework_internal",
        ":lib",
        ":lib_internal",
        ":ops",
        ":protos_all_cc",
        ":test",
        ":test_main",
        ":testlib",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core/kernels:array",
        "//tensorflow/core/kernels:data_flow",
        "//tensorflow/core/kernels:function_ops",
        "//tensorflow/core/kernels:math",
        "//third_party/eigen3",
    ],
)

tf_cc_test(
    name = "ops_remote_fused_graph_ops_test",
    size = "small",
    srcs = ["ops/remote_fused_graph_ops_test.cc"],
    linkstatic = tf_kernel_tests_linkstatic(),
    deps = [
        ":core",
        ":core_cpu",
        ":core_cpu_internal",
        ":framework",
        ":framework_internal",
        ":lib",
        ":lib_internal",
        ":ops",
        ":protos_all_cc",
        ":test",
        ":test_main",
        ":testlib",
        "//tensorflow/core/kernels:remote_fused_graph_ops",
    ],
)

tf_cc_tests(
    name = "ops_tests",
    size = "small",
    srcs = [
        "ops/array_ops_test.cc",
        "ops/candidate_sampling_ops_test.cc",
        "ops/control_flow_ops_test.cc",
        "ops/ctc_ops_test.cc",
        "ops/data_flow_ops_test.cc",
        "ops/functional_ops_test.cc",
        "ops/image_ops_test.cc",
        "ops/io_ops_test.cc",
        "ops/linalg_ops_test.cc",
        "ops/math_ops_test.cc",
        "ops/nn_ops_test.cc",
        "ops/parsing_ops_test.cc",
        "ops/random_ops_test.cc",
        "ops/rnn_ops_test.cc",
        "ops/set_ops_test.cc",
        "ops/shape_function_test.cc",
        "ops/sparse_csr_matrix_ops_test.cc",
        "ops/sparse_ops_test.cc",
        "ops/spectral_ops_test.cc",
        "ops/state_ops_test.cc",
        "ops/string_ops_test.cc",
        "ops/training_ops_test.cc",
    ],
    linkstatic = tf_kernel_tests_linkstatic(),
    deps = [
        ":core",
        ":core_cpu",
        ":core_cpu_internal",
        ":framework",
        ":framework_internal",
        ":lib",
        ":lib_internal",
        ":ops",
        ":protos_all_cc",
        ":test",
        ":test_main",
        ":testlib",
        "//tensorflow/cc:cc_ops",
        "//third_party/eigen3",
    ],
)

tf_cc_test(
    name = "example_example_parser_configuration_test",
    size = "small",
    srcs = ["example/example_parser_configuration_test.cc"],
    data = [":example_parser_configuration_testdata"],
    deps = [
        ":core_cpu",
        ":core_cpu_internal",
        ":direct_session_internal",
        ":example_parser_configuration",
        ":framework",
        ":framework_internal",
        ":lib",
        ":lib_internal",
        ":ops",
        ":protos_all_cc",
        ":test",
        ":test_main",
        ":testlib",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core/kernels:example_parsing_ops",
    ],
)

tf_cc_tests(
    name = "common_runtime_input_colocation_exemption_registry_test",
    size = "small",
    srcs = ["common_runtime/input_colocation_exemption_registry_test.cc"],
    deps = [
        ":core_cpu",
        ":core_cpu_internal",
        ":test",
        ":test_main",
        ":testlib",
    ],
)

tf_cc_tests(
    name = "common_runtime_lower_function_call_test",
    size = "small",
    srcs = ["common_runtime/lower_function_call_op_test.cc"],
    deps = [
        ":all_kernels",
        ":core_cpu",
        ":core_cpu_internal",
        ":direct_session",
        ":framework",
        ":framework_internal",
        ":lib",
        ":test",
        ":test_main",
        ":testlib",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:cc_ops_internal",
        "//tensorflow/cc:client_session",
        "//tensorflow/cc:function_ops",
        "//tensorflow/cc:ops",
        "//tensorflow/cc:resource_variable_ops",
    ],
)

tf_cc_tests(
    name = "common_runtime_lower_if_op_test",
    size = "small",
    srcs = ["common_runtime/lower_if_op_test.cc"],
    deps = [
        ":all_kernels",
        ":core_cpu",
        ":core_cpu_internal",
        ":direct_session",
        ":framework",
        ":framework_internal",
        ":lib",
        ":test",
        ":test_main",
        ":testlib",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:cc_ops_internal",
        "//tensorflow/cc:client_session",
        "//tensorflow/cc:function_ops",
        "//tensorflow/cc:ops",
        "//tensorflow/cc:resource_variable_ops",
    ],
)

tf_cc_tests(
    name = "common_runtime_lower_case_op_test",
    size = "small",
    srcs = ["common_runtime/lower_case_op_test.cc"],
    deps = [
        ":all_kernels",
        ":core_cpu",
        ":core_cpu_internal",
        ":direct_session",
        ":framework",
        ":framework_internal",
        ":lib",
        ":test",
        ":test_main",
        ":testlib",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:cc_ops_internal",
        "//tensorflow/cc:client_session",
        "//tensorflow/cc:function_ops",
        "//tensorflow/cc:ops",
        "//tensorflow/cc:resource_variable_ops",
    ],
)

tf_cc_tests(
    name = "common_runtime_lower_while_op_test",
    size = "small",
    srcs = ["common_runtime/lower_while_op_test.cc"],
    deps = [
        ":all_kernels",
        ":core_cpu",
        ":core_cpu_internal",
        ":direct_session",
        ":framework",
        ":framework_internal",
        ":lib",
        ":test",
        ":test_main",
        ":testlib",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:cc_ops_internal",
        "//tensorflow/cc:client_session",
        "//tensorflow/cc:function_ops",
        "//tensorflow/cc:ops",
        "@com_google_absl//absl/algorithm:container",
    ],
)

tf_cc_tests(
    name = "common_runtime_lower_functional_ops_test",
    size = "small",
    srcs = ["common_runtime/lower_functional_ops_test.cc"],
    deps = [
        ":all_kernels",
        ":core_cpu",
        ":core_cpu_internal",
        ":direct_session",
        ":framework",
        ":framework_internal",
        ":lib",
        ":test",
        ":test_main",
        ":testlib",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:cc_ops_internal",
        "//tensorflow/cc:client_session",
        "//tensorflow/cc:function_ops",
        "//tensorflow/cc:ops",
    ],
)

# Test data
filegroup(
    name = "image_testdata",
    srcs = [
        # PNG data
        "lib/png/testdata/lena_gray.png",
        "lib/png/testdata/lena_rgba.png",
        "lib/png/testdata/lena_palette.png",
        "lib/png/testdata/lena_palette_trns.png",
        # JPEG data
        "lib/jpeg/testdata/jpeg_merge_test1.jpg",
        "lib/jpeg/testdata/jpeg_merge_test1_cmyk.jpg",
        # JPEG data for jpeg benchmark.
        "lib/jpeg/testdata/small.jpg",
        "lib/jpeg/testdata/medium.jpg",
        # Corrupted JPEG files for tests
        "lib/jpeg/testdata/bad_huffman.jpg",
        "lib/jpeg/testdata/corrupt.jpg",
        # -- hand-edited variant: stops at line 0
        "lib/jpeg/testdata/corrupt34_2.jpg",
        # -- hand-edited variant: stops at line 4
        "lib/jpeg/testdata/corrupt34_3.jpg",
        # -- hand-edited variant: stops after a restart marker
        "lib/jpeg/testdata/corrupt34_4.jpg",
        # GIF data
        "lib/gif/testdata/lena.gif",
        "lib/gif/testdata/scan.gif",
        # GIF data with optimization
        "lib/gif/testdata/optimized.gif",
        # BMP data
        "lib/bmp/testdata/lena.bmp",
        # SSIM, PSNR data
        "lib/ssim/testdata/checkerboard1.png",
        "lib/ssim/testdata/checkerboard2.png",
        "lib/ssim/testdata/checkerboard3.png",
        "lib/psnr/testdata/cat_q20.jpg",
        "lib/psnr/testdata/cat_q72.jpg",
        "lib/psnr/testdata/cat_q95.jpg",
    ],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "lmdb_testdata",
    testonly = 1,
    srcs = [
        # A simple key-value store:
        #   0 : 'b'
        #   1 : 'b'
        #    ...
        #   9 : 'b'
        # Which is then overwritten with:
        #   0 : 'a'
        #   1 : 'b'
        #    ...
        #   9 : 'j'
        "lib/lmdb/testdata/data.mdb",
        # LMDB, being a memory-mapped database, uses a different file format on
        # big-endian systems.
        "lib/lmdb/testdata/data_bigendian.mdb",
    ],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "example_parser_configuration_testdata",
    srcs = [
        "example/testdata/parse_example_graph_def.pbtxt",
    ],
)

cc_library(
    name = "cuda_libdevice_path",
    srcs = [
        "//tensorflow/core/platform:legacy_libdevice_srcs",
    ],
    copts = tf_copts(),
    data = tf_additional_libdevice_data(),
    textual_hdrs = ["//tensorflow/core/platform:cuda_libdevice_path.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":lib",
    ] + tf_additional_libdevice_deps(),
)

transitive_hdrs(
    name = "headers",
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        ":core_cpu",
        ":framework",
        ":lib",
        ":protos_all_cc",
        ":stream_executor",
        "//tensorflow/core/platform:platform_strings",
    ],
)

genrule(
    name = "emscripten_proto_config_lite_runtime",
    outs = ["emscripten_proto_config_lite_runtime.asciipb"],
    cmd = tf_genrule_cmd_append_to_srcs("optimize_mode:LITE_RUNTIME"),
    visibility = ["//visibility:private"],
)

# We are keeping the "android" version of tf_android_core_proto_headers. All it does is
# normalize CORE_PROTO_SRCS to generate valid output file names.
tf_portable_proto_library(
    name = "emscripten_proto_lib_no_rtti_lite_runtime",
    config = ":emscripten_proto_config_lite_runtime",
    copts = tf_opts_nortti_if_emscripten(),
    features = tf_features_nomodules_if_emscripten(),
    header_outs = tf_android_core_proto_headers(CORE_PROTO_SRCS) + ["//google/protobuf/any.proto.h"],
    link_full_protobuf = False,
    prefix_dir = "emscripten_proto_no_rtti",
    proto_deps = [
        ":protos_all_cc",
        "@com_google_protobuf//:protobuf",
    ],
    visibility = ["//visibility:public"],
)

# There is currently no need for a full proto version of emscripten tf lib lite.
alias(
    name = "emscripten_lib_lite_no_runtime",
    actual = ":emscripten_tensorflow_lib_lite_nortti_lite_protos_no_runtime",
    visibility = ["//visibility:public"],
)

alias(
    name = "android_srcs_no_runtime",
    actual = ":mobile_srcs_no_runtime",
    visibility = ["//visibility:public"],
)

alias(
    name = "android_srcs_only_runtime",
    actual = ":mobile_srcs_only_runtime",
    visibility = ["//visibility:public"],
)

alias(
    name = "android_srcs",
    actual = ":mobile_srcs",
    visibility = ["//visibility:public"],
)

# Description:
# TensorFlow is a computational framework, primarily for use in machine
# learning applications.
#
# Public targets:
#
# ":protos_all" - exports all core TensorFlow protos
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
# Public mobile targets, e.g. for Android:
#
# filegroup ":android_proto_srcs" - Protos
# filegroup ":android_srcs" - Core sources
# cc_library ":portable_tensorflow_lib" - Native library
# cc_library ":portable_tensorflow_lib_lite" - Native library, without ops,
#   supporting SELECTIVE_REGISTRATION feature.
# portable_proto_library ":portable_proto_lib" (Google-internal)
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
    "if_chromiumos",
    "if_cuda_or_rocm",
    "if_ios",
    "if_mobile",
    "if_not_windows",
    "tf_android_core_proto_headers",
    "tf_cc_test",
    "tf_cc_test_mkl",
    "tf_cc_tests",
    "tf_copts",
    "tf_cuda_library",
    "tf_defines_nortti_if_lite_protos",
    "tf_features_nomodules_if_mobile",
    "tf_gen_op_libs",
    "tf_genrule_cmd_append_to_srcs",
    "tf_opts_nortti_if_lite_protos",
    "tf_portable_full_lite_protos",
    "transitive_hdrs",
)

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "if_nccl")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "tensorflow_opensource_extra_deps")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "tf_cc_test_gpu")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "tf_cc_tests_gpu")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "tf_monitoring_deps")

# For platform specific build config
load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_additional_all_protos",
    "tf_additional_lib_deps",
    "tf_additional_test_deps",
    "tf_jspb_proto_library",
    "tf_kernel_tests_linkstatic",
    "tf_lib_proto_parsing_deps",
    "tf_portable_deps_no_runtime",
    "tf_portable_proto_lib",
    "tf_proto_library",
    "tf_proto_library_cc",
    "tf_protos_all_impl",
    "tf_protos_grappler_impl",
    "tf_protos_profiler_impl",
    "tf_pyclif_proto_library",
)
load(
    "//tensorflow/core/platform:rules_cc.bzl",
    "cc_library",
)
load(
    "//tensorflow/core/platform:build_config_root.bzl",
    "if_dynamic_kernels",
    "if_static",
    "tf_cuda_tests_tags",
)
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")
load("@local_config_tensorrt//:build_defs.bzl", "if_tensorrt")
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
        "//smartass/brain/server/...",
    ],
)

# Export the BUILD file so automated tooling can check licenses
exports_files([
    "BUILD",
    "ops/ops.pbtxt",
])

package_group(
    name = "experimental_access",
    packages = ["//tensorflow/core/common_runtime/..."],
)

# Authorized users go here.
package_group(name = "friends")

# -----------------------------------------------------------------------------
# Public targets

# Protos which are needed for core tensorflow, including on mobile builds.
#
# Note that some protos are in neither additional_core_proto_srcs nor this
# filegroup; e.g.  ones with individual proto_library targets.
# LINT.IfChange
COMMON_PROTO_SRCS = [
    "protobuf/bfc_memory_map.proto",
    "protobuf/config.proto",
    "protobuf/cluster.proto",
    "protobuf/debug.proto",
    "protobuf/device_filters.proto",
    "protobuf/device_properties.proto",
    "protobuf/graph_debug_info.proto",
    "protobuf/queue_runner.proto",
    "protobuf/rewriter_config.proto",
    "protobuf/tensor_bundle.proto",
    "protobuf/saver.proto",
    "protobuf/verifier_config.proto",
]

EXAMPLE_PROTO_SRCS = [
    "//tensorflow/core/example:example.proto",
    "//tensorflow/core/example:feature.proto",
]

UTIL_PROTO_SRCS = [
    "//tensorflow/core/util:event.proto",
    "//tensorflow/core/util:memmapped_file_system.proto",
    "//tensorflow/core/util:saved_tensor_slice.proto",
]

FRAMEWORK_PROTO_SRCS = [
    "//tensorflow/core/framework:allocation_description.proto",
    "//tensorflow/core/framework:api_def.proto",
    "//tensorflow/core/framework:attr_value.proto",
    "//tensorflow/core/framework:cost_graph.proto",
    "//tensorflow/core/framework:device_attributes.proto",
    "//tensorflow/core/framework:function.proto",
    "//tensorflow/core/framework:graph.proto",
    "//tensorflow/core/framework:graph_transfer_info.proto",
    "//tensorflow/core/framework:kernel_def.proto",
    "//tensorflow/core/framework:log_memory.proto",
    "//tensorflow/core/framework:node_def.proto",
    "//tensorflow/core/framework:op_def.proto",
    "//tensorflow/core/framework:reader_base.proto",
    "//tensorflow/core/framework:remote_fused_graph_execute_info.proto",
    "//tensorflow/core/framework:resource_handle.proto",
    "//tensorflow/core/framework:step_stats.proto",
    "//tensorflow/core/framework:summary.proto",
    "//tensorflow/core/framework:tensor.proto",
    "//tensorflow/core/framework:tensor_description.proto",
    "//tensorflow/core/framework:tensor_shape.proto",
    "//tensorflow/core/framework:tensor_slice.proto",
    "//tensorflow/core/framework:types.proto",
    "//tensorflow/core/framework:variable.proto",
    "//tensorflow/core/framework:versions.proto",
]

PROFILER_PROTO_SRCS = [
    "//tensorflow/core/profiler/protobuf:xplane.proto",
    "//tensorflow/core/profiler:profiler_options.proto",
]

ERROR_CODES_PROTO_SRCS = [
    "protobuf/error_codes.proto",
    "//tensorflow/core/lib/core:error_codes.proto",
]
# LINT.ThenChange(//tensorflow/core/portable_proto_config.asciipb)

CORE_PROTO_SRCS = COMMON_PROTO_SRCS + EXAMPLE_PROTO_SRCS + FRAMEWORK_PROTO_SRCS + UTIL_PROTO_SRCS + PROFILER_PROTO_SRCS + ERROR_CODES_PROTO_SRCS

tf_proto_library(
    name = "protos_all",
    srcs = [],
    cc_api_version = 2,
    make_default_target_header_only = True,
    protodeps = [
        ":core_protos",
        ":error_codes_proto_impl",
        "//tensorflow/core/example:protos_all",
        "//tensorflow/core/framework:protos_all",
        "//tensorflow/core/lib/core:error_codes_proto",
        "//tensorflow/core/util:protos_all",
        "//tensorflow/core/util:test_log_proto_impl",
    ],
    visibility = ["//visibility:public"],
)

tf_jspb_proto_library(
    name = "protos_all_jspb_proto",
    visibility = ["//visibility:public"],
    deps = [":protos_all"],
)

alias(
    name = "example_protos",
    actual = "//tensorflow/core/example:example_protos",
    visibility = ["//visibility:public"],
)

java_proto_library(
    name = "example_java_proto",
    visibility = ["//visibility:public"],
    deps = [":example_protos"],
)

cc_library(
    name = "platform_base",
    hdrs = ["//tensorflow/core/platform:base_hdrs"],
    copts = tf_copts(),
    tags = ["avoid_dep"],
    visibility = [":__subpackages__"],
    deps = [
        "//tensorflow/core/platform",
        "//tensorflow/core/platform:byte_order",
        "//tensorflow/core/platform:cord",
        "//tensorflow/core/platform:env_time",
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/platform:macros",
        "//tensorflow/core/platform:threadpool_interface",
        "//tensorflow/core/platform:threadpool_options",
        "//tensorflow/core/platform:types",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/strings",
    ],
)

alias(
    name = "framework_bounds_check",
    actual = "//tensorflow/core/framework:bounds_check",
    visibility = ["//tensorflow/core/kernels:friends"],
)

alias(
    name = "human_readable_json",
    actual = "//tensorflow/core/platform:human_readable_json",
)

# Minimal lib so that tools used for mobile compilation
# don't have to depend on lib/platformlib.
cc_library(
    name = "lib_proto_parsing",
    hdrs = [
        "//tensorflow/core/lib/bfloat16:bfloat16.h",
        "//tensorflow/core/lib/core:legacy_lib_proto_parsing_headers",
        "//tensorflow/core/lib/strings:legacy_lib_proto_parsing_headers",
        "//tensorflow/core/platform:lib_proto_parsing_hdrs",
    ],
    copts = tf_copts(),
    deps = tf_lib_proto_parsing_deps() + [
        ":platform_base",
        "@com_google_absl//absl/strings",
        "@double_conversion//:double-conversion",
        "//tensorflow/core/lib/bfloat16",
        "//tensorflow/core/lib/core:errors",
        "//tensorflow/core/lib/core:stringpiece",
        "//tensorflow/core/lib/core:status",
        "//tensorflow/core/lib/strings:numbers",
        "//tensorflow/core/lib/strings:strcat",
        "//tensorflow/core/platform:platform_port",
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/platform:macros",
        "//tensorflow/core/platform:protobuf",
        "//tensorflow/core/platform",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:stringpiece",
        "//tensorflow/core/platform:tstring",
        "//tensorflow/core/platform:types",
    ],
)

# This build rule (along with :lib_internal, :framework, and
# :framework_internal) purposefully omits the definitions of many declared
# symbols, which are included in //tensorflow:libtensorflow_framework.so. Using
# tf_cc_test and tf_cc_binary will include the necessary symbols.
cc_library(
    name = "lib",
    hdrs = [
        "//tensorflow/core/lib/bfloat16:bfloat16.h",
        "//tensorflow/core/lib/core:legacy_lib_core_headers",
        "//tensorflow/core/lib/gtl:legacy_lib_gtl_headers",
        "//tensorflow/core/lib/hash:legacy_lib_hash_all_headers",
        "//tensorflow/core/lib/histogram:legacy_lib_histogram_all_headers",
        "//tensorflow/core/lib/io:legacy_lib_io_headers",
        "//tensorflow/core/lib/math:math_util.h",
        "//tensorflow/core/lib/monitoring:legacy_lib_monitoring_lib_headers",
        "//tensorflow/core/lib/random:legacy_lib_random_headers",
        "//tensorflow/core/lib/strings:legacy_lib_string_headers",
        "//tensorflow/core/platform:lib_hdrs",
        "//tensorflow/core/util:lib_hdrs",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":lib_internal",
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

alias(
    name = "feature_util",
    actual = "//tensorflow/core/example:feature_util",
    visibility = ["//visibility:public"],
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
        "//tensorflow/core/platform:legacy_test_srcs",
    ],
    hdrs = [
        "//tensorflow/core/lib/core:legacy_lib_core_status_test_util_header",
        "//tensorflow/core/platform:test_hdrs",
        "//tensorflow/core/util:test_hdrs",
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
        "//tensorflow/core/example:feature_util.h",
        "//tensorflow/core/framework:allocator.h",
        "//tensorflow/core/framework:allocator_registry.h",
        "//tensorflow/core/framework:attr_value_util.h",
        "//tensorflow/core/framework:bfloat16.h",
        "//tensorflow/core/framework:bounds_check.h",
        "//tensorflow/core/framework:cancellation.h",
        "//tensorflow/core/framework:collective.h",
        "//tensorflow/core/framework:common_shape_fns.h",
        "//tensorflow/core/framework:control_flow.h",  # TODO(josh11b): Make internal?
        "//tensorflow/core/framework:dataset.h",
        "//tensorflow/core/framework:dataset_stateful_op_whitelist.h",
        "//tensorflow/core/framework:device_base.h",
        "//tensorflow/core/framework:function.h",
        "//tensorflow/core/framework:function_handle_cache.h",
        "//tensorflow/core/framework:graph_def_util.h",
        "//tensorflow/core/framework:graph_to_functiondef.h",
        "//tensorflow/core/framework:kernel_def_builder.h",
        "//tensorflow/core/framework:kernel_def_util.h",
        "//tensorflow/core/framework:kernel_shape_util.h",
        "//tensorflow/core/framework:log_memory.h",
        "//tensorflow/core/framework:logging.h",
        "//tensorflow/core/framework:lookup_interface.h",
        "//tensorflow/core/framework:memory_types.h",
        "//tensorflow/core/framework:node_def_builder.h",
        "//tensorflow/core/framework:node_def_util.h",
        "//tensorflow/core/framework:node_properties.h",
        "//tensorflow/core/framework:numeric_op.h",
        "//tensorflow/core/framework:numeric_types.h",
        "//tensorflow/core/framework:op.h",
        "//tensorflow/core/framework:op_def_builder.h",
        "//tensorflow/core/framework:op_def_util.h",
        "//tensorflow/core/framework:op_kernel.h",
        "//tensorflow/core/framework:ops_util.h",
        "//tensorflow/core/framework:partial_tensor_shape.h",
        "//tensorflow/core/framework:queue_interface.h",
        "//tensorflow/core/framework:reader_interface.h",
        "//tensorflow/core/framework:reader_op_kernel.h",
        "//tensorflow/core/framework:register_types.h",
        "//tensorflow/core/framework:register_types_traits.h",
        "//tensorflow/core/framework:resource_mgr.h",
        "//tensorflow/core/framework:resource_op_kernel.h",
        "//tensorflow/core/framework:selective_registration.h",
        "//tensorflow/core/framework:session_state.h",
        "//tensorflow/core/framework:shape_inference.h",
        "//tensorflow/core/framework:shared_ptr_variant.h",
        "//tensorflow/core/framework:stats_aggregator.h",
        "//tensorflow/core/framework:tensor.h",
        "//tensorflow/core/framework:tensor_shape.h",
        "//tensorflow/core/framework:tensor_slice.h",
        "//tensorflow/core/framework:tensor_types.h",
        "//tensorflow/core/framework:tensor_util.h",
        "//tensorflow/core/framework:thread_factory.h",
        "//tensorflow/core/framework:tracking_allocator.h",
        "//tensorflow/core/framework:type_index.h",
        "//tensorflow/core/framework:type_traits.h",
        "//tensorflow/core/framework:typed_allocator.h",
        "//tensorflow/core/framework:types.h",
        "//tensorflow/core/framework:variant.h",
        "//tensorflow/core/framework:variant_encode_decode.h",
        "//tensorflow/core/framework:variant_op_registry.h",
        "//tensorflow/core/framework:variant_tensor_data.h",
        "//tensorflow/core/util/sparse:framework_group",
        "//tensorflow/core/util:framework_srcs",
        "//tensorflow/core/util:memmapped_file_system_hdrs",
        "//tensorflow/core/public:version.h",
    ] + if_mkl([
        "//tensorflow/core/util:mkl_util_hdrs",
    ]),
    visibility = ["//visibility:public"],
    deps = [
        ":framework_internal",
        "@com_google_absl//absl/base",
    ],
)

# TODO(gonnet): Remove this alias once all users have been moved to the actual target.
alias(
    name = "allocator",
    actual = "//tensorflow/core/framework:allocator",
    visibility = ["//visibility:public"],
)

# TODO(gonnet): Remove this alias once all users have been moved to the actual target.
alias(
    name = "allocator_registry_impl",
    actual = "//tensorflow/core/framework:allocator_registry_impl",
    visibility = ["//visibility:public"],
)

alias(
    name = "overflow",
    actual = "//tensorflow/core/util:overflow",
)

alias(
    name = "exec_on_stall",
    actual = "//tensorflow/core/util:exec_on_stall",
)

alias(
    name = "ptr_util",
    actual = "//tensorflow/core/util:ptr_util",
)

# TODO(gonnet): Remove this alias once all users have been moved to the actual target.
alias(
    name = "reader_base",
    actual = "//tensorflow/core/framework:reader_base",
    visibility = ["//visibility:public"],
)

# TODO(gonnet): Remove this alias once all users have been moved to the actual target.
alias(
    name = "op_gen_lib",
    actual = "//tensorflow/core/framework:op_gen_lib",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "session_options",
    hdrs = ["//tensorflow/core/public:session_options.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":lib",
        ":protos_all_cc",
    ],
)

cc_library(
    name = "framework_lite",
    hdrs = [
        "//tensorflow/core/framework:numeric_types.h",
        "//tensorflow/core/framework:tensor_types.h",
        "//tensorflow/core/framework:type_traits.h",
        "//tensorflow/core/lib/bfloat16:bfloat16.h",
        "//tensorflow/core/platform:framework_lite_hdrs",
        "//tensorflow/core/platform/default:integral_types.h",
        "//tensorflow/core/platform/default:logging.h",
    ],
    visibility = ["//visibility:public"],
    deps =
        [
            "@nsync//:nsync_cpp",
        ] + [
            "//third_party/eigen3",
            "//tensorflow/core/lib/bfloat16",
            "//tensorflow/core/platform:dynamic_annotations",
            "//tensorflow/core/platform:platform_port",
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
        "count_ops",
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
        "special_math_ops",
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
        "tpu_embedding_load_retrieve_ops",
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
        "//tensorflow/core/protobuf/tpu:optimization_parameters_proto_cc",
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
        ":framework",
        ":protos_all_cc",
    ],
)

tf_cc_test(
    name = "ragged_to_dense_util_test",
    srcs = [
        "ops/ragged_to_dense_util_test.cc",
    ],
    deps = [
        ":framework",
        ":protos_all_cc",
        ":ragged_to_dense_util",
        ":test",
        ":testlib",
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
        ":count_ops_op_lib",
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
        ":special_math_ops_op_lib",
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
        ":training_ops_op_lib",
        ":user_ops_op_lib",
        ":word2vec_ops",
        "//tensorflow/c/kernels:bitcast_op_lib",
        "//tensorflow/compiler/mlir/tensorflow:mlir_passthrough_op",
    ] + if_chromiumos(
        [],
        # Non-tpu platforms don't need tpu dependency. It would be best to guard
        # them by if_tpu. But there is no such flag yet.
        [
            ":tpu_configuration_ops_op_lib",
            ":tpu_cross_replica_ops_op_lib",
            ":tpu_embedding_ops_op_lib",
            ":tpu_embedding_load_retrieve_ops_op_lib",
            ":tpu_functional_ops_op_lib",
            ":tpu_heartbeat_ops_op_lib",
            ":tpu_host_compute_ops_op_lib",
            ":tpu_infeed_ops_op_lib",
            ":tpu_outfeed_ops_op_lib",
            ":tpu_ordinal_selector_ops_op_lib",
            ":tpu_replication_ops_op_lib",
        ],
    ) + if_mkl([
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

alias(
    name = "core_cpu",
    actual = "//tensorflow/core/common_runtime:core_cpu",
    visibility = ["//visibility:public"],
)

alias(
    name = "core",
    actual =
        "//tensorflow/core/common_runtime:core",
    visibility = ["//visibility:public"],
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
        "//tensorflow/core/kernels:constant_op",
        "//tensorflow/core/kernels:control_flow_ops",
        "//tensorflow/core/kernels:count_ops",
        "//tensorflow/core/kernels:ctc_ops",
        "//tensorflow/core/kernels:data_flow",
        "//tensorflow/core/kernels:decode_proto_op",
        "//tensorflow/core/kernels:encode_proto_op",
        "//tensorflow/core/kernels:fact_op",
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
        "//tensorflow/core/kernels:quantized_ops",
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
        "//tensorflow/core/kernels:mkl_tmp_bf16_ops",
    ]) + if_cuda_or_rocm([
        "//tensorflow/core/kernels:cudnn_rnn_kernels",
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
    name = "dynamic_kernels_impl",
    visibility = [":__subpackages__"],
    deps = [
        "//tensorflow/core/kernels:sobol_op",
    ],
)

cc_library(
    name = "all_kernels",
    visibility = ["//visibility:public"],
    deps = if_dynamic_kernels(
        [],
        otherwise = [":dynamic_kernels_impl"],
    ) + [
        # TODO(gunan): Split this up and load the resulting kernels dynamically once
        # the dependency issues are resolved.
        ":all_kernels_impl",
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

cc_library(
    name = "distributed_tensorflow_dependencies",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core/distributed_runtime/rpc:grpc_session",
        "//tensorflow/core/kernels:data_service_ops",
    ],
)

cc_library(
    name = "testlib_kernels_impl",
    deps = [
        "//tensorflow/core/kernels:aggregate_ops",
        "//tensorflow/core/kernels:bcast_ops",
        "//tensorflow/core/kernels:cast_op",
        "//tensorflow/core/kernels:constant_op",
        "//tensorflow/core/kernels:identity_op",
        "//tensorflow/core/kernels:random_ops",
        "//tensorflow/core/kernels:reduction_ops",
        "//tensorflow/core/kernels:reshape_op",
    ],
)

# Test support library needed for higher-level (TensorFlow-specific) tests
cc_library(
    name = "testlib",
    testonly = 1,
    srcs = [
        "//tensorflow/core/common_runtime:testlib_srcs",
        "//tensorflow/core/graph:testlib_srcs",
    ],
    hdrs = [
        "//tensorflow/core/common_runtime:testlib_headers",
        "//tensorflow/core/graph:testlib_headers",
        # TODO(josh11b): Drop this once users are depending on
        # kernels:ops_testutil instead.
        "//tensorflow/core/kernels:ops_testutil.h",
        # TODO(gonnet): Stop exporting these headers once users depend on
        # the tensorflow/core/framework:* targets directly.
        "//tensorflow/core/framework:shape_inference_testutil.h",
        "//tensorflow/core/framework:tensor_testutil.h",
        "//tensorflow/core/framework:fake_input.h",
        "//tensorflow/core/framework:function_testlib.h",
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
        ":test",
        ":testlib_ops",
        # TODO(gunan): resolve dependency issues and load these kernels dynamically.
        ":testlib_kernels_impl",
        "//tensorflow/cc:scope",
        "//tensorflow/core/framework:fake_input",
        "//tensorflow/core/framework:function_testlib",
        "//tensorflow/core/framework:shape_inference_testutil",
        "//tensorflow/core/framework:tensor_testutil",
        "//tensorflow/core/kernels:ops_testutil",
        "//tensorflow/core/kernels:ops_util",
    ],
)

alias(
    name = "testlib_ops",
    testonly = 1,
    actual =
        "//tensorflow/core/common_runtime:testlib_ops",
)

# This is a link-only library to provide a DirectSession
# implementation of the Session interface.
tf_cuda_library(
    name = "direct_session",
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core/common_runtime:direct_session_internal",
    ],
    alwayslink = 1,
)

# -----------------------------------------------------------------------------
# MKL targets
alias(
    name = "mkl_graph_util",
    actual = "//tensorflow/core/graph:mkl_graph_util",
)

# -----------------------------------------------------------------------------
# Public Android targets

# List of protos we want on android
filegroup(
    name = "android_proto_srcs",
    srcs = CORE_PROTO_SRCS,
    visibility = ["//visibility:public"],
)

# Sources required to build the TensorFlow framework without the runtime on
# mobile platforms. This is essentially the sources required to build
# tensorflow/core/framework:tensor without using granular targets.
filegroup(
    name = "mobile_srcs_no_runtime",
    srcs = [
        "//tensorflow/compiler/jit:mobile_srcs_no_runtime",
        "//tensorflow/core/example:mobile_srcs_no_runtime",
        "//tensorflow/core/framework:attr_value_proto_text_srcs",
        "//tensorflow/core/framework:mobile_srcs_no_runtime",
        "//tensorflow/core/lib/bfloat16:mobile_srcs_no_runtime",
        "//tensorflow/core/lib/core:mobile_srcs_no_runtime",
        "//tensorflow/core/lib/gtl:mobile_srcs_no_runtime",
        "//tensorflow/core/lib/hash:mobile_srcs_no_runtime",
        "//tensorflow/core/lib/strings:mobile_srcs_no_runtime",
        "//tensorflow/core/platform:mobile_srcs_no_runtime",
        "//tensorflow/core/public:mobile_srcs_no_runtime",
        "//tensorflow/core/util:mobile_srcs_no_runtime",
        "//tensorflow/core/util/ctc:android_srcs",
    ] + glob(
        [
            "client/**/*.cc",
        ],
        exclude = [
            "**/*test.*",
            "**/*testutil*",
            "**/*testlib*",
            "**/*main.cc",
        ],
    ),
    visibility = ["//visibility:private"],
)

# Sources required to build the TensorFlow framework with runtime on
# mobile platforms without granular targets. It is assumed that the source
# files in tensorflow/core:mobile_srcs_no_runtime have been compiled
# separately and are linked in as a dependency.
filegroup(
    name = "mobile_srcs_only_runtime",
    srcs = [
        # Sources for which we do not yet have granular targets.
        "//tensorflow/c/eager:srcs",
        "//tensorflow/c/experimental/saved_model/core:mobile_srcs_only_runtime",
        "//tensorflow/c:srcs",
        "//tensorflow/core/common_runtime:mobile_srcs_only_runtime",
        "//tensorflow/core/common_runtime/eager:srcs",
        "//tensorflow/core/framework:mobile_srcs_only_runtime",
        "//tensorflow/core/graph:mobile_srcs_only_runtime",
        "//tensorflow/core/kernels:android_srcs",
        "//tensorflow/core/lib/io:mobile_srcs_only_runtime",
        "//tensorflow/core/profiler:mobile_srcs",
        "//tensorflow/core/public:mobile_srcs_only_runtime",
        "//tensorflow/core/util/sparse:mobile_srcs_only_runtime",
        "//tensorflow/core/util/tensor_bundle:android_srcs",
        "//tensorflow/core/util:mobile_srcs_only_runtime",

        # Sources for which we already have granular targets.
        "//tensorflow/core/lib/core:mobile_srcs_only_runtime",
        "//tensorflow/core/lib/gtl:mobile_srcs_only_runtime",
        "//tensorflow/core/lib/hash:mobile_srcs_only_runtime",
        "//tensorflow/core/lib/histogram:mobile_srcs_only_runtime",
        "//tensorflow/core/lib/math:mobile_srcs_only_runtime",
        "//tensorflow/core/lib/monitoring:mobile_srcs_only_runtime",
        "//tensorflow/core/lib/random:mobile_srcs_only_runtime",
        "//tensorflow/core/lib/strings:mobile_srcs_only_runtime",
        "//tensorflow/core/platform:mobile_srcs_only_runtime",
    ] + glob(
        [
            "lib/wav/*.cc",
            "lib/wav/*.h",
        ],
        exclude = [
            "**/*test.*",
            "**/*testutil*",
            "**/*testlib*",
            "**/*main.cc",
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

alias(
    name = "android_srcs",
    actual = ":mobile_srcs",
    visibility = ["//visibility:public"],
)

# Native library support for mobile applications.  Does not contain
# operators, use :portable_tensorflow_lib if you want full operator
# support.
#
# If you just need TensorFlow types, e.g. Tensors, use
# :portable_tensorflow_lib_lite_no_runtime.
#
# These targets link against either full or lite protos depending
# on the bazel define TENSORFLOW_PROTOS=full/lite respectively.
#
# Compiles to a trivial library on non-mobile to prevent irrelevant
# build errors. If not building this e.g. as part of an android_binary,
# a command such as the following must be used:
# bazel build -c opt tensorflow/core:android_tensorflow_lib \
# --define=TENSORFLOW_PROTOS=lite \
# --crosstool_top=//external:android/crosstool \
# --cpu=armeabi-v7a \
# --host_crosstool_top=@bazel_tools//tools/cpp:toolchain
cc_library(
    name = "portable_tensorflow_lib_lite",
    srcs = if_mobile([":mobile_srcs"]),
    copts = tf_copts(android_optimization_level_override = None) + tf_opts_nortti_if_lite_protos() + if_ios(["-Os"]),
    defines = ["SUPPORT_SELECTIVE_REGISTRATION"] + if_chromiumos(["IS_MOBILE_PLATFORM"]) + tf_defines_nortti_if_lite_protos(),
    linkopts = if_android(["-lz"]) + if_ios(["-lz"]),
    tags = [
        "manual",
        "notap",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core/util:stats_calculator_portable",
        "//tensorflow/core:mobile_additional_lib_deps",
    ] + tf_portable_proto_lib() + tf_portable_deps_no_runtime(),
    alwayslink = 1,
)

alias(
    name = "android_tensorflow_lib_lite",
    actual = ":portable_tensorflow_lib_lite",
    visibility = ["//visibility:public"],
)

alias(
    name = "android_tensorflow_lib_lite_nortti",
    actual = ":portable_tensorflow_lib_lite",
    visibility = ["//visibility:public"],
)

alias(
    name = "android_tensorflow_lib_lite_nortti_lite_protos",
    actual = ":portable_tensorflow_lib_lite",
    visibility = ["//visibility:public"],
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

alias(
    name = "ios_tensorflow_lib_lite",
    actual = ":portable_tensorflow_lib_lite",
    visibility = ["//visibility:public"],
)

# Full TensorFlow library with operator support. Use this unless reducing
# binary size (by packaging a reduced operator set) is a concern.
alias(
    name = "android_tensorflow_lib",
    actual = ":portable_tensorflow_lib",
    visibility = ["//visibility:public"],
)

alias(
    name = "ios_tensorflow_lib",
    actual = ":portable_tensorflow_lib",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "portable_tensorflow_lib",
    srcs = if_mobile([":portable_op_registrations_and_gradients"]),
    copts = tf_copts() + tf_opts_nortti_if_lite_protos() + if_ios(["-Os"]),
    features = tf_features_nomodules_if_mobile(),
    tags = [
        "manual",
        "notap",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":portable_tensorflow_lib_lite",
        ":protos_all_cc_impl",
        "//tensorflow/core/kernels:portable_tensorflow_kernels",
        "//third_party/eigen3",
        "@com_google_protobuf//:protobuf",
    ],
    alwayslink = 1,
)

alias(
    name = "android_op_registrations_and_gradients",
    actual = ":portable_op_registrations_and_gradients",
    visibility = ["//visibility:public"],
)

filegroup(
    name = "portable_op_registrations_and_gradients",
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
    testonly = 1,
    # TODO(andrewharp/nhua):
    # make more test-related sources portable e.g. "//tensorflow/core/platform:test.cc",
    srcs = tf_portable_full_lite_protos(
        full = [
            "//tensorflow/core/framework:android_test_hdrs",
            "//tensorflow/core/framework:android_test_srcs",
            "//tensorflow/core/platform:android_test_srcs",
            "//tensorflow/core/util:android_test_srcs",
        ],
        lite = [
            "//tensorflow/core/framework:android_test_hdrs",
            "//tensorflow/core/framework:android_test_srcs_no_core",
            "//tensorflow/core/platform:android_test_srcs",
            "//tensorflow/core/util:android_test_srcs",
        ],
    ),
    visibility = ["//visibility:public"],
)

# This is like android_test_srcs, minus the things that are already in mobile_srcs.
filegroup(
    name = "android_test_srcs_no_core",
    testonly = 1,
    srcs = [
        "//tensorflow/core/framework:android_test_hdrs",
        "//tensorflow/core/framework:android_test_srcs_no_core",
        "//tensorflow/core/platform:android_test_srcs",
        "//tensorflow/core/util:android_test_srcs",
    ],
    visibility = ["//visibility:public"],
)

# Portable library providing testing functionality for TensorFlow.
alias(
    name = "android_tensorflow_test_lib",
    actual = ":portable_tensorflow_test_lib",
    visibility = ["//visibility:public"],
)

alias(
    name = "ios_tensorflow_test_lib",
    actual = ":portable_tensorflow_test_lib",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "portable_tensorflow_test_lib",
    testonly = 1,
    srcs = if_mobile([
        ":android_test_srcs_no_core",
    ]),
    hdrs = [
        "//tensorflow/core/framework:android_test_hdrs",
        "//tensorflow/core/util:android_test_hdrs",
    ],
    copts = tf_copts(android_optimization_level_override = None) + if_ios(["-Os"]),
    features = tf_features_nomodules_if_mobile() + tf_opts_nortti_if_lite_protos(),
    tags = [
        "manual",
        "notap",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":portable_tensorflow_lib",
        ":protos_all_cc",
        "//tensorflow/core/kernels:portable_tensorflow_kernels",
        "//tensorflow/core/platform/default/build_config:gtest",
        "//third_party/eigen3",
    ],
)

# -----------------------------------------------------------------------------
# Libraries with GPU facilities that are useful for writing kernels.
cc_library(
    name = "gpu_lib",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core/common_runtime/gpu:gpu_lib",
    ],
)

alias(
    name = "gpu_headers_lib",
    actual =
        "//tensorflow/core/common_runtime/gpu:gpu_headers_lib",
    visibility = ["//visibility:public"],
)

alias(
    name = "cuda",
    actual = "//tensorflow/core/common_runtime/gpu:cuda",
    visibility = ["//visibility:public"],
)

# -----------------------------------------------------------------------------
# Clif-related proto libraries.

# The following targets will be moved to core/example. The aliases are only temporary
# since moving existing users will require several CLs over several projects.
[
    [
        alias(
            name = "example_%s_pyclif%s" % (proto_name, target_suffix),
            actual = "//tensorflow/core/example:%s_pyclif%s" % (proto_name, target_suffix),
            visibility = ["//visibility:public"],
        )
        for target_suffix in [
            "",
            "_pb2",
        ]
    ]
    for proto_name in [
        "example",
        "feature",
    ]
]

# The following targets will be moved to core/protobuf. The aliases are only temporary
# since moving existing users will require several CLs over several projects.
[
    [
        alias(
            name = "protobuf_%s_pyclif%s" % (proto_name, target_suffix),
            actual = ":protobuf/%s_pyclif%s" % (proto_name, target_suffix),
            visibility = ["//visibility:public"],
        )
        for target_suffix in [
            "",
            "_pb2",
        ]
    ] + [
        tf_pyclif_proto_library(
            name = "protobuf/%s_pyclif" % proto_name,
            proto_lib = ":protos_all",
            proto_srcfile = "protobuf/%s.proto" % proto_name,
            visibility = ["//visibility:public"],
        ),
    ]
    for proto_name in [
        "config",
        "device_properties",
        "graph_debug_info",
        "meta_graph",
        "saved_model",
    ]
]

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

filegroup(
    name = "lib_internal_private_headers",
    srcs = [
        "//tensorflow/core/framework:resource_handle.h",
        "//tensorflow/core/platform:legacy_lib_internal_headers",
        "//tensorflow/core/platform:lib_internal_private_hdrs",
        "//tensorflow/core/lib/bfloat16:bfloat16.h",
        "//tensorflow/core/lib/core:legacy_lib_core_all_headers",
        "//tensorflow/core/lib/gtl:legacy_lib_gtl_all_headers",
        "//tensorflow/core/lib/histogram:legacy_lib_histogram_all_headers",
        "//tensorflow/core/lib/hash:legacy_lib_hash_all_headers",
        "//tensorflow/core/lib/monitoring:legacy_lib_monitoring_all_headers",
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
    ),
    visibility = ["//visibility:private"],
)

filegroup(
    name = "lib_internal_public_headers",
    srcs = [
        "lib/wav/wav_io.h",
        "//tensorflow/core/lib/core:legacy_lib_internal_core_headers",
        "//tensorflow/core/lib/gtl:legacy_lib_internal_public_gtl_headers",
        "//tensorflow/core/lib/hash:legacy_lib_internal_public_headers",
        "//tensorflow/core/lib/io:legacy_lib_internal_public_headers",
        "//tensorflow/core/lib/monitoring:legacy_lib_monitoring_lib_internal_public_headers",
        "//tensorflow/core/lib/random:legacy_lib_internal_public_random_headers",
        "//tensorflow/core/lib/strings:legacy_lib_internal_public_string_headers",
        "//tensorflow/core/platform:legacy_platform_lib_hdrs",
        "//tensorflow/core/platform:lib_internal_public_hdrs",
        "//tensorflow/core/util:lib_internal_public_hdrs",
    ],
    visibility = ["//visibility:private"],
)

cc_library(
    name = "lib_internal",
    srcs = [":lib_internal_private_headers"],
    hdrs = [":lib_internal_public_headers"],
    copts = tf_copts(),
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
        "@com_google_absl//absl/meta:type_traits",
        "@com_google_absl//absl/strings",
        "//third_party/eigen3",
        "@com_google_absl//absl/base:core_headers",
        "//tensorflow/core/platform/default/build_config:platformlib",
    ] + if_static([":lib_internal_impl"]),
)

# Until we can ditch config=monolithic on windows, we have to provide an always
# headers only library for pybind rules to depend on.
cc_library(
    name = "lib_headers_for_pybind",
    srcs = [":lib_internal_private_headers"],
    hdrs = [":lib_internal_public_headers"],
    copts = tf_copts(),
    linkopts = select({
        "//tensorflow:freebsd": [],
        "//tensorflow:windows": [],
        "//tensorflow:android": [],
        "//conditions:default": [
            "-ldl",
            "-lpthread",
        ],
    }),
    visibility = ["//tensorflow/python:__pkg__"],
    deps = tf_additional_lib_deps() + [
        "@com_google_absl//absl/meta:type_traits",
        "@com_google_absl//absl/strings",
        "//third_party/eigen3",
        "@com_google_absl//absl/base:core_headers",
        "//tensorflow/core/platform/default/build_config:platformlib",
    ],
)

cc_library(
    name = "lib_internal_impl",
    srcs = [
        ":lib_internal_private_headers",
        "//tensorflow/core/platform:legacy_lib_internal_srcs",
    ] + glob(
        [
            "lib/**/*.cc",
        ],
        exclude = [
            "**/*test*",
            "framework/variant.cc",
            "lib/gif/**/*",
            "lib/jpeg/**/*",
            "lib/png/**/*",
        ],
    ),
    hdrs = [":lib_internal_public_headers"],
    copts = tf_copts(),
    deps = tf_additional_lib_deps() + [
        ":core_stringpiece",
        ":lib_proto_parsing",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "//third_party/eigen3",
        "//tensorflow/core/lib/bfloat16",
        "//tensorflow/core/lib/core:arena",
        "//tensorflow/core/lib/core:bitmap",
        "//tensorflow/core/lib/core:blocking_counter",
        "//tensorflow/core/lib/core:bits",
        "//tensorflow/core/lib/core:coding",
        "//tensorflow/core/lib/core:errors",
        "//tensorflow/core/lib/core:notification",
        "//tensorflow/core/lib/core:raw_coding",
        "//tensorflow/core/lib/core:refcount",
        "//tensorflow/core/lib/core:status",
        "//tensorflow/core/lib/core:stringpiece",
        "//tensorflow/core/lib/core:threadpool",
        "//tensorflow/core/lib/core:threadpool_interface",
        "//tensorflow/core/lib/core:threadpool_options",
        "//tensorflow/core/lib/gtl:array_slice",
        "//tensorflow/core/lib/gtl:cleanup",
        "//tensorflow/core/lib/gtl:compactptrset",
        "//tensorflow/core/lib/gtl:edit_distance",
        "//tensorflow/core/lib/gtl:flatmap",
        "//tensorflow/core/lib/gtl:flatrep",
        "//tensorflow/core/lib/gtl:flatset",
        "//tensorflow/core/lib/gtl:inlined_vector",
        "//tensorflow/core/lib/gtl:int_type",
        "//tensorflow/core/lib/gtl:iterator_range",
        "//tensorflow/core/lib/gtl:manual_constructor",
        "//tensorflow/core/lib/gtl:map_util",
        "//tensorflow/core/lib/gtl:optional",
        "//tensorflow/core/lib/gtl:priority_queue_util",
        "//tensorflow/core/lib/gtl:top_n",
        "//tensorflow/core/lib/hash:crc32c",
        "//tensorflow/core/lib/hash",
        "//tensorflow/core/lib/histogram",
        "//tensorflow/core/lib/io:block",
        "//tensorflow/core/lib/io:buffered_inputstream",
        "//tensorflow/core/lib/io:compression",
        "//tensorflow/core/lib/io:inputbuffer",
        "//tensorflow/core/lib/io:inputstream_interface",
        "//tensorflow/core/lib/io:iterator",
        "//tensorflow/core/lib/io:path",
        "//tensorflow/core/lib/io:proto_encode_helper",
        "//tensorflow/core/lib/io:random_inputstream",
        "//tensorflow/core/lib/io:record_reader",
        "//tensorflow/core/lib/io:record_writer",
        "//tensorflow/core/lib/io:snappy_inputbuffer",
        "//tensorflow/core/lib/io:snappy_outputbuffer",
        "//tensorflow/core/lib/io:table",
        "//tensorflow/core/lib/io:table_options",
        "//tensorflow/core/lib/io:zlib_compression_options",
        "//tensorflow/core/lib/io:zlib_inputstream",
        "//tensorflow/core/lib/io:zlib_outputbuffer",
        "//tensorflow/core/lib/math:math_util",
        "//tensorflow/core/lib/monitoring:collected_metrics",
        "//tensorflow/core/lib/monitoring:collection_registry",
        "//tensorflow/core/lib/monitoring:counter",
        "//tensorflow/core/lib/monitoring:gauge",
        "//tensorflow/core/lib/monitoring:metric_def",
        "//tensorflow/core/lib/monitoring:mobile_counter",
        "//tensorflow/core/lib/monitoring:mobile_gauge",
        "//tensorflow/core/lib/monitoring:mobile_percentile_sampler",
        "//tensorflow/core/lib/monitoring:mobile_sampler",
        "//tensorflow/core/lib/monitoring:percentile_sampler",
        "//tensorflow/core/lib/monitoring:sampler",
        "//tensorflow/core/lib/monitoring:timed",
        "//tensorflow/core/lib/random:exact_uniform_int",
        "//tensorflow/core/lib/random:philox",
        "//tensorflow/core/lib/random:philox_random",
        "//tensorflow/core/lib/random",
        "//tensorflow/core/lib/random:weighted_picker",
        "//tensorflow/core/lib/strings:base64",
        "//tensorflow/core/lib/strings:numbers",
        "//tensorflow/core/lib/strings:ordered_code",
        "//tensorflow/core/lib/strings:proto_serialization",
        "//tensorflow/core/lib/strings:proto_text_util",
        "//tensorflow/core/lib/strings:scanner",
        "//tensorflow/core/lib/strings:str_util",
        "//tensorflow/core/lib/strings:strcat",
        "//tensorflow/core/lib/strings:stringprintf",
        "//tensorflow/core/platform:abi",
        "//tensorflow/core/platform:base64",
        "//tensorflow/core/platform:blocking_counter",
        "//tensorflow/core/platform:casts",
        "//tensorflow/core/platform:coding",
        "//tensorflow/core/platform:context",
        "//tensorflow/core/platform:cord",
        "//tensorflow/core/platform:cpu_feature_guard",
        "//tensorflow/core/platform:denormal",
        "//tensorflow/core/platform:dynamic_annotations",
        "//tensorflow/core/platform:env",
        "//tensorflow/core/platform:error",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:file_statistics",
        "//tensorflow/core/platform:fingerprint",
        "//tensorflow/core/platform:hash",
        "//tensorflow/core/platform:load_library",
        "//tensorflow/core/platform:logger",
        "//tensorflow/core/platform:mutex",
        "//tensorflow/core/platform:notification",
        "//tensorflow/core/platform:net",
        "//tensorflow/core/platform:null_file_system",
        "//tensorflow/core/platform:numbers",
        "//tensorflow/core/platform:path",
        "//tensorflow/core/platform:platform_port",
        "//tensorflow/core/platform:platform_strings",
        "//tensorflow/core/platform:prefetch",
        "//tensorflow/core/platform:protobuf_internal",
        "//tensorflow/core/platform:regexp",
        "//tensorflow/core/platform:resource",
        "//tensorflow/core/platform:scanner",
        "//tensorflow/core/platform:setround",
        "//tensorflow/core/platform:stacktrace",
        "//tensorflow/core/platform:stacktrace_handler",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:strcat",
        "//tensorflow/core/platform:stringpiece",
        "//tensorflow/core/platform:stringprintf",
        "//tensorflow/core/platform:strong_hash",
        "//tensorflow/core/platform:str_util",
        "//tensorflow/core/platform:subprocess",
        "//tensorflow/core/platform:tensor_coding",
        "//tensorflow/core/platform:thread_annotations",
        "//tensorflow/core/platform:threadpool_interface",
        "//tensorflow/core/platform:threadpool_options",
        "//tensorflow/core/platform:tracing",
        "//tensorflow/core/platform:tstring",
        "//tensorflow/core/platform:unbounded_work_queue",
        "//tensorflow/core/platform/default/build_config:platformlib",
        "//tensorflow/core/util:env_var",
        "//tensorflow/core/util:reporter",  # TODO(gunan): REMOVE as soon as cc_shared_library is supported.
        "@snappy",
        "@zlib",
        "@double_conversion//:double-conversion",
        "@com_google_protobuf//:protobuf",
    ] + tf_protos_all_impl() + tf_protos_grappler_impl() + tf_protos_profiler_impl() + tf_monitoring_deps(),
    # Alwayslink causes a cc_binary to "always link" in the
    # srcs for a given cc_library, even if they are unreferenced, see:
    # https://docs.bazel.build/versions/master/be/c-cpp.html#cc_library.alwayslink
    # tensorflow/core:lib_internal_impl has alwayslink set so that libtensorflow_framework.so
    # contains all of the srcs within lib_internal_impl.
    # NOTE(bmzhao): As we refactor tensorflow's BUILD files to more granular targets,
    # we've removed srcs from core:lib_internal_impl, and have added deps on these
    # granular targets instead. Since alwayslink doesn't apply to these deps,
    # we will sometimes need to add alwayslink to the granular libraries instead.
    # Ideally, this will be resolved once bazel has full cc_shared_library support.
    alwayslink = 1,
)

cc_library(
    name = "gif_internal",
    srcs = [
        "lib/gif/gif_io.cc",
        "//tensorflow/core/platform:gif_hdrs",
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
        "//tensorflow/core/platform:jpeg_hdrs",
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

alias(
    name = "png_internal",
    actual = "//tensorflow/core/lib/png:png_io",
)

alias(
    name = "android_png_internal",
    actual = "//tensorflow/core/lib/png:png_io",
)

cc_library(
    name = "tflite_portable_logging",
    hdrs = [
        "//tensorflow/core/lib/bfloat16:bfloat16.h",
        "//tensorflow/core/platform:tflite_portable_logging_hdrs",
        "//tensorflow/core/platform/default:integral_types.h",
        "//tensorflow/core/platform/default:logging.h",
    ],
    copts = tf_copts(),
    linkopts = select({
        "//tensorflow:freebsd": [],
        "//tensorflow:windows": [],
        "//conditions:default": [
            "-ldl",
        ],
    }),
    visibility = ["//visibility:public"],
    deps = [
        ":platform_base",
        "//tensorflow/core/platform:logging",
    ],
)

cc_library(
    name = "android_jpeg_internal",
    srcs = if_android([
        "lib/jpeg/jpeg_handle.cc",
        "lib/jpeg/jpeg_mem.cc",
        "//tensorflow/core/platform:jpeg_hdrs",
    ]),
    hdrs = [
        "lib/jpeg/jpeg_handle.h",
        "lib/jpeg/jpeg_mem.h",
        "//tensorflow/core/lib/bfloat16:bfloat16.h",
        "//tensorflow/core/lib/core:legacy_lib_core_stringpiece_header",
        "//tensorflow/core/platform:jpeg_internal_hdrs",
        "//tensorflow/core/platform/default:integral_types.h",
        "//tensorflow/core/platform/default:logging.h",
    ],
    copts = tf_copts(),
    linkopts = ["-ldl"],
    deps = [
        ":core_stringpiece",
        "//tensorflow/core/platform:dynamic_annotations",
        "//tensorflow/core/platform:jpeg",
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/platform:stringpiece",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "android_gif_internal",
    srcs = if_android([
        "lib/gif/gif_io.cc",
        "//tensorflow/core/platform:gif_hdrs",
    ]),
    hdrs = [
        "lib/gif/gif_io.h",
        "//tensorflow/core/lib/bfloat16:bfloat16.h",
        "//tensorflow/core/lib/core:legacy_lib_core_stringpiece_header",
        "//tensorflow/core/lib/gtl:legacy_android_gif_internal_headers",
        "//tensorflow/core/platform:gif_internal_hdrs",
        "//tensorflow/core/platform/default:integral_types.h",
        "//tensorflow/core/platform/default:logging.h",
    ],
    copts = tf_copts(),
    linkopts = ["-ldl"],
    deps = [
        "//tensorflow/core/lib/strings:numbers",
        "//tensorflow/core/lib/strings:strcat",
        "//tensorflow/core/platform:dynamic_annotations",
        "//tensorflow/core/platform:gif",
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/platform:numbers",
        "//tensorflow/core/platform:strcat",
        "//tensorflow/core/platform:stringpiece",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
    ],
)

tf_proto_library(
    name = "error_codes_proto_impl",
    srcs = ["protobuf/error_codes.proto"],
    cc_api_version = 2,
    make_default_target_header_only = True,
)

tf_proto_library(
    name = "core_protos",
    srcs = COMMON_PROTO_SRCS + [
        # Protos which are not needed on mobile builds, but should be included
        # in protos_all.
        #
        # Note that some protos are in neither core_proto_srcs nor this
        # filegroup; e.g. ones with individual proto_library targets.
        "protobuf/control_flow.proto",
        # TODO(ebrevdo): Re-enable once CriticalSection is in core.
        # "protobuf/critical_section.proto",
        "protobuf/data/experimental/snapshot.proto",
        "protobuf/debug_event.proto",
        "protobuf/meta_graph.proto",
        "protobuf/named_tensor.proto",
        "protobuf/remote_tensor_handle.proto",
        "protobuf/saved_model.proto",
        "protobuf/saved_object_graph.proto",
        "protobuf/struct.proto",
        "protobuf/tensorflow_server.proto",
        "protobuf/trackable_object_graph.proto",
        "protobuf/transport_options.proto",
    ],
    cc_api_version = 2,
    make_default_target_header_only = True,
    protodeps = [
        ":error_codes_proto_impl",
        "//tensorflow/core/example:protos_all",
        "//tensorflow/core/framework:protos_all",
        "//tensorflow/core/lib/core:error_codes_proto",
        "//tensorflow/core/profiler/protobuf:xplane_proto",
        "//tensorflow/core/profiler:profiler_options_proto",
        "//tensorflow/core/util:protos_all",
        "//tensorflow/core/util:test_log_proto_impl",
    ],
    visibility = ["//visibility:private"],
)

alias(
    name = "error_codes_proto_cc",
    actual = "//tensorflow/core/lib/core:error_codes_proto_cc",
)

alias(
    name = "version_lib",
    actual = "//tensorflow/core/util:version_info",
)

filegroup(
    name = "framework_internal_private_headers",
    srcs = [
        "//tensorflow/core/example:feature_util.h",
        "//tensorflow/core/framework:framework_internal_private_hdrs",
        "//tensorflow/core/graph:framework_internal_private_headers",
        "//tensorflow/core/util:framework_internal_private_hdrs",
        "//tensorflow/core/util:memmapped_file_system_hdrs",
        "//tensorflow/core/util/sparse:framework_internal_private_headers_group",
    ],
    visibility = ["//visibility:private"],
)

filegroup(
    name = "framework_internal_public_headers",
    srcs = [
        "//tensorflow/core/framework:metrics.h",
        "//tensorflow/core/framework:model.h",
        "//tensorflow/core/framework:op_segment.h",
        "//tensorflow/core/framework:rendezvous.h",  # only needed for tests
        "//tensorflow/core/framework:resource_var.h",
        "//tensorflow/core/framework:run_handler.h",
        "//tensorflow/core/framework:run_handler_util.h",
        "//tensorflow/core/framework:shared_ptr_variant.h",
        "//tensorflow/core/framework:tensor_reference.h",
        "//tensorflow/core/framework:tracking_allocator.h",  # only needed for tests
        "//tensorflow/core/framework:variant.h",
        "//tensorflow/core/util:framework_internal_public_hdrs",
    ],
    visibility = ["//visibility:private"],
)

tf_cuda_library(
    name = "framework_internal",
    srcs = [":framework_internal_private_headers"],
    hdrs = [":framework_internal_public_headers"],
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

tf_cuda_library(
    name = "framework_internal_impl",
    srcs = [
        ":framework_internal_private_headers",
        "//tensorflow/core/framework:framework_internal_impl_srcs",
        "//tensorflow/core/graph:framework_internal_impl_srcs",
        "//tensorflow/core/util:framework_internal_impl_srcs",
        "//tensorflow/core/util:memmapped_file_system_srcs",
        "//tensorflow/core/util/sparse:framework_internal_impl_group",
    ] + glob(
        [
        ],
        exclude = [
            "**/*test*",
            "**/*main.cc",
        ],
    ),
    hdrs = [":framework_internal_public_headers"],
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
        ":feature_util",
        ":lib",
        ":lib_internal",
        ":protos_all_cc",
        ":version_lib",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "//third_party/eigen3",
        "//tensorflow/core/example:feature_util",
        "//tensorflow/core/framework:allocator",
        "//tensorflow/core/framework:allocator_registry_impl",
        "//tensorflow/core/framework:attr_value_proto_text",
        "//tensorflow/core/framework:attr_value_util",
        "//tensorflow/core/framework:bfloat16",
        "//tensorflow/core/framework:common_shape_fns",
        "//tensorflow/core/framework:kernel_shape_util",
        "//tensorflow/core/framework:node_def_util",
        "//tensorflow/core/framework:node_properties",
        "//tensorflow/core/framework:numeric_types",
        "//tensorflow/core/framework:op",
        "//tensorflow/core/framework:op_def_builder",
        "//tensorflow/core/framework:op_def_util",
        "//tensorflow/core/framework:resource_handle",
        "//tensorflow/core/framework:selective_registration",
        "//tensorflow/core/framework:shape_inference",
        "//tensorflow/core/framework:tensor",
        "//tensorflow/core/framework:tensor_shape",
        "//tensorflow/core/kernels:bounds_check",
        "//tensorflow/core/platform/default/build_config:platformlib",
        "//tensorflow/core/profiler/lib:annotated_traceme",
        "//tensorflow/core/profiler/lib:traceme",
        "//tensorflow/core/util:einsum_op_util",
        "//tensorflow/core/util:padding",
        "//tensorflow/core/util:port",
        "//tensorflow/core/util:stats_calculator_portable",
        "//tensorflow/core/util:tensor_format",
        "//tensorflow/compiler/jit:common",
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
        "//tensorflow/core/framework:reader_base",
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

alias(
    name = "stream_executor",
    actual = "//tensorflow/core/platform:stream_executor",
)

# Like stream_executor library, but compiles without --config=cuda
# and does not include any cuda dependencies.
alias(
    name = "stream_executor_no_cuda",
    actual = "//tensorflow/core/platform:stream_executor_no_cuda",
    visibility = ["//visibility:public"],
)

alias(
    name = "cuda_device_functions",
    actual = "//tensorflow/core/util:gpu_device_functions",
    visibility = ["//visibility:public"],
)

# Library containing all of the graph construction code that is
# independent of the runtime.
tf_cuda_library(
    name = "graph",
    srcs = ["//tensorflow/core/graph:graph_srcs"],
    hdrs = ["//tensorflow/core/graph:graph_headers"],
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

filegroup(
    name = "core_cpu_base_headers",
    srcs = [
        "//tensorflow/core/common_runtime:core_cpu_base_headers",
    ],
)

tf_cuda_library(
    name = "core_cpu_base",
    hdrs = [
        ":core_cpu_base_headers",
        "//tensorflow/core/public:session.h",
    ],
    copts = tf_copts(),
    deps = ["//tensorflow/core/common_runtime:core_cpu_base_no_ops"] + if_static([
        ":function_ops_op_lib",
        ":functional_grad",
        ":functional_ops_op_lib",
        "//tensorflow/core/kernels:bounds_check",
        "//tensorflow/core/kernels:required",
    ]),
    alwayslink = 1,
)

alias(
    name = "core_cpu_impl",
    actual =
        "//tensorflow/core/common_runtime:core_cpu_impl",
)

alias(
    name = "core_cpu_lib",
    actual =
        "//tensorflow/core/common_runtime:core_cpu_lib",
)

alias(
    name = "core_cpu_internal",
    actual =
        "//tensorflow/core/common_runtime:core_cpu_internal",
)

alias(
    name = "regexp_internal",
    actual =
        "//tensorflow/core/platform:regexp",
    visibility = [
        "//tensorflow/compiler:__subpackages__",
        "//tensorflow/core/kernels:__subpackages__",
        "//tensorflow/core/profiler:__subpackages__",
        "//tensorflow/stream_executor:__subpackages__",
    ],
)

alias(
    name = "direct_session_internal",
    actual =
        "//tensorflow/core/common_runtime:direct_session_internal",
)

alias(
    name = "example_parser_configuration",
    actual = "//tensorflow/core/example:example_parser_configuration",
    visibility = ["//visibility:public"],
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

alias(
    name = "gpu_runtime",
    actual =
        "//tensorflow/core/common_runtime/gpu:gpu_runtime",
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

# TODO(gonnet): Remove this alias once all users have been moved to the actual target.
alias(
    name = "tensor_testutil",
    actual = "//tensorflow/core/framework:tensor_testutil",
)

# TODO(gonnet): Remove this alias once all users have been moved to the actual target.
alias(
    name = "shape_inference_testutil",
    actual = "//tensorflow/core/framework:shape_inference_testutil",
)

# Main program for tests
alias(
    name = "test_main",
    actual = "//tensorflow/core/platform:test_main",
    visibility = ["//tensorflow:internal"],
)

test_suite(
    name = "low_level_tests",
    tests = [
        ":low_level_library_tests",
        "//tensorflow/core/platform:low_level_library_tests",
    ],
)

tf_cc_tests(
    name = "low_level_library_tests",
    size = "small",
    srcs = [
        "lib/wav/wav_io_test.cc",
        "//tensorflow/core/lib/core:legacy_lib_core_all_tests",
        "//tensorflow/core/lib/gtl:legacy_lib_gtl_tests",
        "//tensorflow/core/lib/hash:legacy_lib_hash_all_tests",
        "//tensorflow/core/lib/histogram:legacy_lib_histogram_all_tests",
        "//tensorflow/core/lib/io:legacy_lib_io_all_tests",
        "//tensorflow/core/lib/math:math_util_test.cc",
        "//tensorflow/core/lib/monitoring:collection_registry_test.cc",
        "//tensorflow/core/lib/monitoring:counter_test.cc",
        "//tensorflow/core/lib/monitoring:gauge_test.cc",
        "//tensorflow/core/lib/monitoring:metric_def_test.cc",
        "//tensorflow/core/lib/monitoring:percentile_sampler_test.cc",
        "//tensorflow/core/lib/monitoring:sampler_test.cc",
        "//tensorflow/core/lib/random:legacy_lib_random_tests",
        "//tensorflow/core/lib/strings:legacy_low_level_library_tests",
    ],
    create_named_test_suite = True,
    deps = [
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
        "@zlib",
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

test_suite(
    name = "platform_tests",
    tests = [
        "//tensorflow/core/platform:abi_test",
        "//tensorflow/core/platform:env_test",
        "//tensorflow/core/platform:fake_python_env_test",
        "//tensorflow/core/platform:file_system_test",
        "//tensorflow/core/platform:numa_test",
        "//tensorflow/core/platform:platform_strings_test",
        "//tensorflow/core/platform:rocm_rocdl_path_test",
        "//tensorflow/core/platform:setround_test",
        "//tensorflow/core/platform:unbounded_work_queue_test",
        "//tensorflow/core/platform:vmodule_test",
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
    srcs = ["//tensorflow/core/framework:op_gen_lib_test.cc"],
    deps = [
        ":protos_all_cc",
        ":test",
        ":test_main",
        "//tensorflow/core/framework:op_gen_lib",
    ],
)

test_suite(
    name = "higher_level_tests",
    tests = [
        ":core_higher_level_tests",
        "//tensorflow/core/framework:higher_level_tests",
        "//tensorflow/core/util:higher_level_tests",
    ],
)

tf_cc_tests(
    name = "core_higher_level_tests",
    size = "small",
    srcs = [
        "//tensorflow/core/example:feature_util_test.cc",
        "//tensorflow/core/graph:algorithm_test.cc",
        "//tensorflow/core/graph:control_flow_test.cc",
        "//tensorflow/core/graph:edgeset_test.cc",
        "//tensorflow/core/graph:graph_def_builder_test.cc",
        "//tensorflow/core/graph:graph_partition_test.cc",
        "//tensorflow/core/graph:graph_test.cc",
        "//tensorflow/core/graph:node_builder_test.cc",
        "//tensorflow/core/graph:optimizer_cse_test.cc",
        "//tensorflow/core/graph:subgraph_test.cc",
        "//tensorflow/core/graph:tensor_id_test.cc",
        "//tensorflow/core/graph:validate_test.cc",
        "//tensorflow/core/util/sparse:higher_level_tests_group",
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
        "//tensorflow/cc:ops",
        "//tensorflow/cc:scope",
        "//tensorflow/cc:sendrecv_ops",
        "//tensorflow/cc:while_loop",
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

tf_cc_test_mkl(
    name = "mkl_related_tests",
    size = "small",
    srcs = [
        "//tensorflow/core/util:mkl_util_test_srcs",
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
        "//tensorflow/core/kernels:mkl_tmp_bf16_ops",
    ]),
)

tf_cc_tests_gpu(
    name = "gpu_related_tests",
    size = "small",
    srcs = glob(["user_ops/**/*_test.cc"]),
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
    name = "variant_op_copy_test",
    size = "small",
    srcs = ["//tensorflow/core/framework:variant_op_copy_test.cc"],
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
    name = "framework_run_handler_util_test",
    size = "small",
    srcs = ["//tensorflow/core/framework:run_handler_util_test.cc"],
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
    srcs = ["//tensorflow/core/framework:run_handler_test.cc"],
    linkstatic = tf_kernel_tests_linkstatic(),
    deps = [
        ":core_cpu",
        ":direct_session_internal",
        ":framework_internal",
        ":lib",
        ":lib_internal",
        ":protos_all_cc",
        ":test",
        ":test_main",
        ":testlib",
        "//tensorflow/core/framework:tensor_testutil",
        "//tensorflow/core/kernels:cwise_op",
        "//tensorflow/core/kernels:matmul_op",
        "//third_party/eigen3",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/synchronization",
    ],
)

tf_cc_test(
    name = "framework_op_segment_test",
    size = "small",
    srcs = ["//tensorflow/core/framework:op_segment_test.cc"],
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

tf_cc_test(
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

# Test data
filegroup(
    name = "image_testdata",
    srcs = [
        # PNG data
        "//tensorflow/core/lib/png:testdata",
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

alias(
    name = "cuda_libdevice_path",
    actual = "//tensorflow/core/platform:cuda_libdevice_path",
)

# Normalize CORE_PROTO_SRCS to generate valid output file names.
PORTABLE_PROTO_HEADERS_OUT = tf_android_core_proto_headers(CORE_PROTO_SRCS) + [
    "//google/protobuf/any.proto.h",
]

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

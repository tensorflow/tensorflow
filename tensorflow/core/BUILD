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
#     //third_party/tensorflow/lite/kernels/shim: Template for writing ops supporting TF & TF Lite.
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
# cc_library ":portable_tensorflow_lib" - Native library
# cc_library ":portable_tensorflow_lib_lite" - Native library, without ops,
#   supporting SELECTIVE_REGISTRATION feature.
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
    "if_android",
    "if_cuda_or_rocm",
    "if_google",
    "if_ios",
    "if_libtpu",
    "if_mobile",
    "if_nccl",
    "tf_cc_test",
    "tf_cc_test_mkl",
    "tf_cc_tests",
    "tf_copts",
    "tf_cuda_library",
    "tf_defines_nortti_if_lite_protos",
    "tf_features_nomodules_if_mobile",
    "tf_opts_nortti_if_lite_protos",
    "transitive_hdrs",
)
load("//tensorflow:tensorflow.default.bzl", "cc_header_only_library", "filegroup", "get_compatible_with_portable", "tensorflow_opensource_extra_deps", "tf_monitoring_framework_deps", "tf_selective_registration_deps")

# For platform specific build config
load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_additional_lib_deps",
    "tf_additional_test_deps",
    "tf_dtensor_tpu_dependencies",
    "tf_jspb_proto_library",
    "tf_lib_proto_parsing_deps",
    "tf_portable_deps_no_runtime",
    "tf_portable_proto_lib",
    "tf_proto_library",
    "tf_protos_grappler_impl",
    "tf_tpu_dependencies",
)
load(
    "//tensorflow/core/platform:rules_cc.bzl",
    "cc_library",
)
load(
    "//tensorflow/core/platform:build_config_root.bzl",
    "if_dynamic_kernels",
    "if_static",
)
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")
load("@local_config_tensorrt//:build_defs.bzl", "if_tensorrt")
load(
    "//third_party/mkl:build_defs.bzl",
    "if_mkl",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        ":dependency_whitelist",
        "//tensorflow:internal",
        "//tensorflow_models:__subpackages__",
    ],
    features = if_google([
        "-layering_check",
        "-parse_headers",
    ]),
    licenses = ["notice"],
)

package_group(
    name = "dependency_whitelist",
    packages = [
        "//learning/freud/topic_models/tensorflow/...",
        "//learning/infra/mira/...",
        "//perftools/accelerators/xprof/api/...",
        "//quality/webanswers/brain/tokenization/custom_tf_ops/kernels/...",
        "//smartass/brain/server/...",
    ],
)

# Export the BUILD file so automated tooling can check licenses
exports_files([
    "BUILD",
])

# Authorized users go here.
package_group(
    name = "friends",
    packages = if_google([
        "//learning/brain/...",
        "//tensorflow/...",
        "@tf_runtime//...",
        "//third_party/tf_runtime_google/...",
    ]),
)

# -----------------------------------------------------------------------------
# Public targets

tf_proto_library(
    name = "protos_all",
    srcs = [],
    cc_api_version = 2,
    create_go_proto = False,
    make_default_target_header_only = True,
    protodeps = [
        "//tensorflow/core/example:protos_all",
        "//tensorflow/core/framework:protos_all",
        "//tensorflow/core/lib/core:error_codes_proto",
        "//tensorflow/core/profiler/protobuf:xplane_proto",
        "//tensorflow/core/profiler:profiler_options_proto",
        "//tensorflow/core/protobuf:error_codes_proto_impl",
        "//tensorflow/core/protobuf:for_core_protos",
        "//tensorflow/core/util:protos_all",
        "//tensorflow/core/util:test_log_proto",
        "//tensorflow/core/grappler/costs:op_performance_data",
        "//tensorflow/tsl/profiler/protobuf:profiler_options_proto",
        "//tensorflow/tsl/profiler/protobuf:xplane_proto",
        "//tensorflow/tsl/protobuf:coordination_config_proto",
        "//tensorflow/tsl/protobuf:distributed_runtime_payloads_proto",
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
    hdrs = [
        "//tensorflow/core/platform:base_hdrs",
        "//tensorflow/tsl/platform:base_hdrs",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = tf_copts(),
    tags = ["avoid_dep"],
    deps = [
        "//tensorflow/core/platform",
        "//tensorflow/core/platform:byte_order",
        "//tensorflow/core/platform:cord",
        "//tensorflow/core/platform:env_time",
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/platform:macros",
        "//tensorflow/core/platform:stack_frame",
        "//tensorflow/core/platform:threadpool_interface",
        "//tensorflow/core/platform:threadpool_options",
        "//tensorflow/core/platform:types",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/strings",
    ],
)

# Minimal lib so that tools used for mobile compilation
# don't have to depend on lib/platformlib.
cc_library(
    name = "lib_proto_parsing",
    hdrs = [
        "//tensorflow/core/lib/core:legacy_lib_proto_parsing_headers",
        "//tensorflow/core/lib/strings:legacy_lib_proto_parsing_headers",
        "//tensorflow/core/platform:lib_proto_parsing_hdrs",
        "//tensorflow/tsl/platform:lib_proto_parsing_hdrs",
    ],
    copts = tf_copts(),
    deps = tf_lib_proto_parsing_deps() + [
        ":platform_base",
        "@com_google_absl//absl/strings",
        "@double_conversion//:double-conversion",
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
        # TODO(rmlarsen): Remove bfloat16.h once dependency in third_party/swift is updated.
        "//tensorflow/core/lib/bfloat16:bfloat16.h",
        "//tensorflow/core/lib/core:legacy_lib_core_headers",
        "//tensorflow/tsl/lib/core:legacy_lib_core_headers",
        "//tensorflow/core/lib/gtl:legacy_lib_gtl_headers",
        "//tensorflow/core/lib/hash:legacy_lib_hash_all_headers",
        "//tensorflow/core/lib/histogram:legacy_lib_histogram_all_headers",
        "//tensorflow/core/lib/io:legacy_lib_io_headers",
        "//tensorflow/core/lib/math:math_util.h",
        "//tensorflow/tsl/lib/math:math_util.h",
        "//tensorflow/core/lib/monitoring:legacy_lib_monitoring_lib_headers",
        "//tensorflow/core/lib/random:legacy_lib_random_headers",
        "//tensorflow/core/lib/strings:legacy_lib_string_headers",
        "//tensorflow/core/platform:lib_hdrs",
        "//tensorflow/core/util:lib_hdrs",
        "//tensorflow/tsl/lib/io:legacy_lib_io_headers",
        "//tensorflow/tsl/platform:lib_hdrs",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":lib_internal",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/types:optional",
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
    hdrs = [
        "//tensorflow/core/lib/core:legacy_lib_core_status_test_util_header",
        "//tensorflow/core/platform:test_hdrs",
        "//tensorflow/core/util:test_hdrs",
        "//tensorflow/tsl/platform:test_hdrs",
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
        "//tensorflow/tsl/platform/default/build_config:gtest",
        "//tensorflow/core/kernels:required",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_benchmark",
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
        "//tensorflow/core/framework:dataset_stateful_op_allowlist.h",
        "//tensorflow/core/framework:device.h",
        "//tensorflow/core/framework:device_base.h",
        "//tensorflow/core/framework:device_factory.h",
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
        "//tensorflow/core/framework:metrics.h",
        "//tensorflow/core/framework:node_def_builder.h",
        "//tensorflow/core/framework:node_def_util.h",
        "//tensorflow/core/framework:node_properties.h",
        "//tensorflow/core/framework:numeric_op.h",
        "//tensorflow/core/framework:numeric_types.h",
        "//tensorflow/core/framework:op.h",
        "//tensorflow/core/framework:op_def_builder.h",
        "//tensorflow/core/framework:op_def_util.h",
        "//tensorflow/core/framework:op_kernel.h",
        "//tensorflow/core/framework:op_requires.h",
        "//tensorflow/core/framework:ops_util.h",
        "//tensorflow/core/framework:partial_tensor_shape.h",
        "//tensorflow/core/framework:queue_interface.h",
        "//tensorflow/core/framework:reader_interface.h",
        "//tensorflow/core/framework:reader_op_kernel.h",
        "//tensorflow/core/framework:register_types.h",
        "//tensorflow/core/framework:register_types_traits.h",
        "//tensorflow/core/framework/registration:options.h",
        "//tensorflow/core/framework:resource_mgr.h",
        "//tensorflow/core/framework:resource_handle.h",
        "//tensorflow/core/framework:resource_op_kernel.h",
        "//tensorflow/core/framework:rng_alg.h",
        "//tensorflow/core/framework/registration:registration.h",
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
        "//tensorflow/core/util:determinism_hdr",
        "//tensorflow/core/public:version.h",
        "//tensorflow/lite/kernels/shim:tf_headers",
        "//tensorflow/tsl/framework:allocator_hdrs",
        "//tensorflow/tsl/framework:cancellation_hdrs",
        "//tensorflow/tsl/framework:device_type.h",
    ] + if_mkl([
        "//tensorflow/core/util:mkl_util_hdrs",
    ]),
    visibility = ["//visibility:public"],
    deps = [
        ":framework_internal",
        "@com_google_absl//absl/base",
    ],
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
    name = "framework_types_hdr",
    hdrs = [
        "//tensorflow/core/framework:types.h",
        "//tensorflow/tsl/framework:device_type.h",
    ],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "framework_lite",
    hdrs = [
        "//tensorflow/core/framework:numeric_types.h",
        "//tensorflow/core/framework:tensor_types.h",
        "//tensorflow/core/framework:type_traits.h",
        "//tensorflow/core/platform:framework_lite_hdrs",
        "//tensorflow/tsl/framework:numeric_types.h",
        "//tensorflow/tsl/framework:type_traits.h",
        "//tensorflow/tsl/platform:framework_lite_hdrs",
        "//tensorflow/tsl/platform/default:integral_types.h",
        "//tensorflow/tsl/platform/default:logging.h",
    ],
    visibility = ["//visibility:public"],
    deps =
        # copybara:uncomment_begin
        # if_mobile(["@nsync//:nsync_cpp"]) +
        # copybara:uncomment_end_uncomment_begin
        # ["@nsync//:nsync_cpp"] +
        # copybara:comment_end
        [
            "//third_party/eigen3",
            "//tensorflow/core/platform:dynamic_annotations",
            "//tensorflow/core/platform:platform_port",
            "//tensorflow/core/platform:thread_annotations",
            "//tensorflow/core/platform:mutex",
            "//tensorflow/tsl/platform/default/build_config:minimal",
            "//tensorflow/tsl/framework:fixedpoint_types",
            "//tensorflow/core/platform:types",
        ],
)

# One target for all user ops.
alias(
    name = "user_ops_op_lib",
    actual = "//tensorflow/core/user_ops:user_ops_op_lib",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "ops",
    visibility = ["//visibility:public"],
    deps = [
        ":user_ops_op_lib",
        "//tensorflow/c/kernels:bitcast_op_lib",
        "//tensorflow/c/kernels:histogram_summary_op_lib",
        "//tensorflow/c/kernels:merge_summary_op_lib",
        "//tensorflow/c/kernels:summary_op_lib",
        "//tensorflow/core/ops:ops",
        "//tensorflow/dtensor/cc:dtensor_ops",
    ] + select({
        # Non-tpu platforms don't need tpu dependency.
        "//tensorflow:chromiumos": [],
        "//tensorflow:fuchsia": [],
        "//conditions:default": [
            "//tensorflow/core/tpu/ops",
            "//tensorflow/dtensor/cc:dtensor_tpu_ops",
        ],
    }) + if_tensorrt([
        "//tensorflow/compiler/tf2tensorrt:trt_engine_resource_ops_op_lib",
        "//tensorflow/compiler/tf2tensorrt:trt_op_libs",
    ]) + if_libtpu(
        if_false = ["//tensorflow/compiler/mlir/tensorflow:mlir_passthrough_op"],
        if_true = [],
    ),
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
        "//tensorflow/c/kernels:histogram_summary_op",
        "//tensorflow/c/kernels:merge_summary_op",
        "//tensorflow/c/kernels:summary_op",
        "//tensorflow/core/kernels:array",
        "//tensorflow/core/kernels:audio",
        "//tensorflow/core/kernels:batch_kernels",
        "//tensorflow/core/kernels:bincount_op",
        "//tensorflow/core/kernels:boosted_trees_ops",
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
        "//tensorflow/core/kernels:composite_tensor_ops",
        "//tensorflow/core/kernels:fact_op",
        "//tensorflow/core/kernels:fake_quant_ops",
        "//tensorflow/core/kernels:filesystem_ops",
        "//tensorflow/core/kernels:function_ops",
        "//tensorflow/core/kernels:functional_ops",
        "//tensorflow/core/kernels:grappler",
        "//tensorflow/core/kernels:histogram_op",
        "//tensorflow/core/kernels:io",
        "//tensorflow/core/kernels:isotonic_regression_op",
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
        "//tensorflow/core/kernels:random_index_shuffle_ops",
        "//tensorflow/core/kernels:random_ops",
        "//tensorflow/core/kernels:stateful_random_ops",
        "//tensorflow/core/kernels:random_binomial_op",
        "//tensorflow/core/kernels:random_poisson_op",
        "//tensorflow/core/kernels:required",
        "//tensorflow/core/kernels:resource_variable_ops",
        "//tensorflow/core/kernels:rnn_ops",
        "//tensorflow/core/kernels:scoped_allocator_ops",
        "//tensorflow/core/kernels:sdca_ops",
        "//tensorflow/core/kernels:searchsorted_op",
        "//tensorflow/core/kernels:set_kernels",
        "//tensorflow/core/kernels:sparse",
        "//tensorflow/core/kernels:state",
        "//tensorflow/core/kernels:stateless_random_ops",
        "//tensorflow/core/kernels:stateless_random_gamma_op",
        "//tensorflow/core/kernels:stateless_shuffle",
        "//tensorflow/core/kernels:string",
        "//tensorflow/core/kernels:summary_kernels",
        "//tensorflow/core/kernels:training_ops",
        "//tensorflow/core/kernels/uniform_quant_ops:kernels",
        "//tensorflow/core/kernels:word2vec_kernels",
        "//tensorflow/core/kernels/linalg:linalg",
        "//tensorflow/core/kernels/image:image",
        "//tensorflow/core/kernels/sparse:kernels",
    ] + if_mkl([
        "//tensorflow/core/kernels/mkl:mkl_aggregate_ops",
        "//tensorflow/core/kernels/mkl:mkl_concat_op",
        "//tensorflow/core/kernels/mkl:mkl_dequantize_op",
        "//tensorflow/core/kernels/mkl:mkl_conv_op",
        "//tensorflow/core/kernels/mkl:mkl_cwise_ops_common",
        "//tensorflow/core/kernels/mkl:mkl_fused_batch_norm_op",
        "//tensorflow/core/kernels/mkl:mkl_identity_op",
        "//tensorflow/core/kernels/mkl:mkl_input_conversion_op",
        "//tensorflow/core/kernels/mkl:mkl_layer_norm_op",
        "//tensorflow/core/kernels/mkl:mkl_lrn_op",
        "//tensorflow/core/kernels/mkl:mkl_pooling_ops",
        "//tensorflow/core/kernels/mkl:mkl_qmatmul_op",
        "//tensorflow/core/kernels/mkl:mkl_requantize_ops",
        "//tensorflow/core/kernels/mkl:mkl_quantize_op",
        "//tensorflow/core/kernels/mkl:mkl_relu_op",
        "//tensorflow/core/kernels/mkl:mkl_reshape_op",
        "//tensorflow/core/kernels/mkl:mkl_slice_op",
        "//tensorflow/core/kernels/mkl:mkl_softmax_op",
        "//tensorflow/core/kernels/mkl:mkl_swish_op",
        "//tensorflow/core/kernels/mkl:mkl_fused_mish_op",
        "//tensorflow/core/kernels/mkl:mkl_transpose_op",
        "//tensorflow/core/kernels/mkl:mkl_batch_matmul_op",
        "//tensorflow/core/kernels/mkl:mkl_einsum_op",
        "//tensorflow/core/kernels/mkl:mkl_matmul_op",
        "//tensorflow/core/kernels/mkl:mkl_tfconv_op",
        "//tensorflow/core/kernels/mkl:mkl_tmp_bf16_ops",
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
    ]) + tf_tpu_dependencies() + tf_dtensor_tpu_dependencies(),
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
        "//tensorflow/core/kernels:map_kernels",
        "//tensorflow/core/kernels:tensor_map",
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
        "//tensorflow/tsl/platform/default/build_config:tensorflow_platform_specific",
    ],
)

cc_library(
    name = "distributed_tensorflow_dependencies",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core/distributed_runtime/rpc:grpc_session",
        "//tensorflow/core/kernels:data_service_ops",
        "//tensorflow/distribute/experimental/rpc/kernels:rpc_ops",
    ],
)

cc_library(
    name = "testlib_kernels_impl",
    deps = [
        "//tensorflow/core/kernels:aggregate_ops",
        "//tensorflow/core/kernels:bcast_ops",
        "//tensorflow/core/kernels:cast_op",
        "//tensorflow/core/kernels:constant_op",
        "//tensorflow/core/kernels:host_constant_op",
        "//tensorflow/core/kernels:identity_op",
        "//tensorflow/core/kernels:random_ops",
        "//tensorflow/core/kernels:reduction_ops",
        "//tensorflow/core/kernels:reshape_op",
    ],
)

# Test support library needed for higher-level (TensorFlow-specific) tests
tf_cuda_library(
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
        # TODO(gunan): resolve dependency issues and load these kernels dynamically.
        ":testlib_kernels_impl",
        "//tensorflow/cc:scope",
        "//tensorflow/core/common_runtime:testlib_ops",
        "//tensorflow/core/framework:fake_input",
        "//tensorflow/core/framework:function_testlib",
        "//tensorflow/core/framework:shape_inference_testutil",
        "//tensorflow/core/framework:tensor_testutil",
        "//tensorflow/core/kernels:ops_testutil",
        "//tensorflow/core/kernels:ops_util",
    ] + if_mkl(["//tensorflow/core/graph:mkl_graph_util"]),
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
# Public Android targets

# Sources required to build the TensorFlow framework without the runtime on
# mobile platforms. This is essentially the sources required to build
# tensorflow/core/framework:tensor without using granular targets.
filegroup(
    name = "mobile_srcs_no_runtime",
    srcs = [
        "//tensorflow/compiler/jit:mobile_srcs_no_runtime",
        "//tensorflow/core/activity_watcher:activity_watcher_headers",
        "//tensorflow/core/example:mobile_srcs_no_runtime",
        "//tensorflow/core/framework:attr_value_proto_text_srcs",
        "//tensorflow/core/framework:mobile_srcs_no_runtime",
        "//tensorflow/core/lib/core:mobile_srcs_no_runtime",
        "//tensorflow/core/lib/gtl:mobile_srcs_no_runtime",
        "//tensorflow/core/lib/hash:mobile_srcs_no_runtime",
        "//tensorflow/core/lib/strings:mobile_srcs_no_runtime",
        "//tensorflow/core/platform:mobile_srcs_no_runtime",
        "//tensorflow/core/profiler:mobile_srcs_no_runtime",
        "//tensorflow/core/public:mobile_srcs_no_runtime",
        "//tensorflow/core/util:mobile_srcs_no_runtime",
        "//tensorflow/core/util/ctc:mobile_srcs",
        "//tensorflow/tsl/framework:mobile_srcs_no_runtime",
        "//tensorflow/tsl/framework/fixedpoint:mobile_srcs_no_runtime",
        "//tensorflow/tsl/platform:mobile_srcs_no_runtime",
    ],
    visibility = ["//visibility:public"],
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
        # StreamExecutor C API is currently not supported for mobile.
        # Including just the header for SP_Stream reference in kernels C API.
        "//tensorflow/c/experimental/stream_executor:headers",
        "//tensorflow/c:srcs",
        "//tensorflow/core/common_runtime:mobile_srcs_only_runtime",
        "//tensorflow/core/common_runtime/eager:srcs",
        "//tensorflow/core/framework:mobile_srcs_only_runtime",
        "//tensorflow/core/graph:mobile_srcs_only_runtime",
        "//tensorflow/core/kernels:mobile_srcs",
        "//tensorflow/core/lib/io:mobile_srcs_only_runtime",
        "//tensorflow/tsl/lib/io:mobile_srcs_only_runtime",
        "//tensorflow/core/nccl:mobile_srcs",
        "//tensorflow/core/profiler:mobile_srcs_only_runtime",
        "//tensorflow/core/public:mobile_srcs_only_runtime",
        "//tensorflow/core/util/sparse:mobile_srcs_only_runtime",
        "//tensorflow/core/util/tensor_bundle:mobile_srcs",
        "//tensorflow/core/util:mobile_srcs_only_runtime",

        # Sources for which we already have granular targets.
        "//tensorflow/core/lib/core:mobile_srcs_only_runtime",
        "//tensorflow/tsl/lib/core:mobile_srcs_only_runtime",
        "//tensorflow/core/lib/gtl:mobile_srcs_only_runtime",
        "//tensorflow/core/lib/hash:mobile_srcs_only_runtime",
        "//tensorflow/core/lib/histogram:mobile_srcs_only_runtime",
        "//tensorflow/core/lib/math:mobile_srcs_only_runtime",
        "//tensorflow/tsl/lib/math:mobile_srcs_only_runtime",
        "//tensorflow/core/lib/monitoring:mobile_srcs_only_runtime",
        "//tensorflow/core/lib/random:mobile_srcs_only_runtime",
        "//tensorflow/core/lib/strings:mobile_srcs_only_runtime",
        "//tensorflow/core/platform:mobile_srcs_only_runtime",
        "//tensorflow/tsl/platform:mobile_srcs_only_runtime",
        "//tensorflow/tsl/framework:mobile_srcs_only_runtime",
        "//tensorflow/tsl/framework/fixedpoint:mobile_srcs_only_runtime",

        # Wav io files
        "//tensorflow/core/lib/wav:wav_io.h",
        "//tensorflow/core/lib/wav:wav_io.cc",
    ],
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

# All the aliases for stuff under ops/
# Once the dependencies move to the real targets, remove the aliases here!

[
    alias(
        name = "%s" % (name,),
        actual = "//tensorflow/core/ops:%s" % (name,),
        visibility = ["//visibility:public"],
    )
    for name in [
        "array_grad",
        "array_ops_op_lib",
        "audio_ops_op_lib",
        "batch_ops_op_lib",
        "bitwise_ops_op_lib",
        "boosted_trees_ops_op_lib",
        "candidate_sampling_ops_op_lib",
        "checkpoint_ops_op_lib",
        "clustering_ops_op_lib",
        "collective_ops_op_lib",
        "control_flow_ops_op_lib",
        "count_ops_op_lib",
        "ctc_ops_op_lib",
        "cudnn_rnn_ops_op_lib",
        "data_flow_ops_op_lib",
        "dataset_ops_op_lib",
        "debug_ops_op_lib",
        "decode_proto_ops_op_lib",
        "encode_proto_ops_op_lib",
        "experimental_dataset_ops_op_lib",
        "filesystem_ops_op_lib",
        "function_ops_op_lib",
        "functional_grad",
        "functional_ops_op_lib",
        "image_ops_op_lib",
        "io_ops_op_lib",
        "linalg_ops_op_lib",
        "list_ops_op_lib",
        "logging_ops_op_lib",
        "lookup_ops_op_lib",
        "manip_ops_op_lib",
        "map_ops_op_lib",
        "math_grad",
        "math_ops_op_lib",
        "mkl_array_ops_op_lib",
        "mkl_nn_ops_op_lib",
        "nccl_ops_op_lib",
        "nn_grad",
        "nn_ops_op_lib",
        "no_op_op_lib",
        "parsing_ops_op_lib",
        "portable_op_registrations_and_gradients",
        "ragged_array_ops_op_lib",
        "ragged_conversion_ops_op_lib",
        "ragged_math_ops_op_lib",
        "ragged_ops",
        "composite_tensor_ops_op_lib",
        "random_grad_op_lib",
        "random_index_shuffle_ops_op_lib",
        "random_ops_op_lib",
        "resource_variable_ops_op_lib",
        "risc_ops_op_lib",
        "rnn_ops_op_lib",
        "scoped_allocator_ops_op_lib",
        "script_ops_op_lib",
        "sdca_ops_op_lib",
        "sendrecv_ops_op_lib",
        "set_ops_op_lib",
        "sparse_csr_matrix_ops_op_lib",
        "sparse_ops_op_lib",
        "special_math_ops_op_lib",
        "spectral_ops_op_lib",
        "state_ops_op_lib",
        "stateful_random_ops_op_lib",
        "stateless_random_ops_op_lib",
        "stateless_random_ops_v2_op_lib",
        "string_ops_op_lib",
        "summary_ops_op_lib",
        "tpu_configuration_ops_op_lib",
        "tpu_cross_replica_ops_op_lib",
        "tpu_embedding_ops_op_lib",
        "tpu_embedding_load_retrieve_ops_op_lib",
        "tpu_functional_ops_op_lib",
        "tpu_heartbeat_ops_op_lib",
        "tpu_host_compute_ops_op_lib",
        "tpu_infeed_ops_op_lib",
        "tpu_outfeed_ops_op_lib",
        "tpu_ordinal_selector_ops_op_lib",
        "tpu_replication_ops_op_lib",
        "tpu_sharding_util_ops_op_lib",
        "training_ops_op_lib",
        "uniform_quant_ops_op_lib",
        "word2vec_ops",
    ]
]

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
# bazel build -c opt tensorflow/core:portable_tensorflow_lib \
# --define=TENSORFLOW_PROTOS=lite \
# --crosstool_top=//external:android/crosstool \
# --cpu=armeabi-v7a \
# --host_crosstool_top=@bazel_tools//tools/cpp:toolchain
cc_library(
    name = "portable_tensorflow_lib_lite",
    srcs = if_mobile([":mobile_srcs"]),
    copts = tf_copts(android_optimization_level_override = None) + tf_opts_nortti_if_lite_protos() + if_ios(["-Os"]),
    defines = ["SUPPORT_SELECTIVE_REGISTRATION"] + select({
                  "//tensorflow:chromiumos": ["IS_MOBILE_PLATFORM"],
                  "//tensorflow:fuchsia": ["IS_MOBILE_PLATFORM"],
                  "//conditions:default": [],
              }) +
              tf_defines_nortti_if_lite_protos(),
    # Using private headers from other libraries violates layering check.
    features = ["-layering_check"],
    linkopts = if_android(["-lz"]) + if_ios(["-lz"]),
    tags = [
        "manual",
        "notap",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core/platform:resource",
        "//tensorflow/core/util:stats_calculator_portable",
        "//tensorflow/core:mobile_additional_lib_deps",
    ] + tf_portable_proto_lib() + tf_portable_deps_no_runtime(),
    alwayslink = 1,
)

# copybara:uncomment_begin(google-only)
# cc_library(
#     name = "portable_tensorflow_lib_lite_no_runtime",
#     srcs = if_mobile([":mobile_srcs_no_runtime"]),
#     copts = tf_copts(android_optimization_level_override = None) + tf_opts_nortti_if_lite_protos() + if_ios(["-Os"]),
#     defines = [
#         "SUPPORT_SELECTIVE_REGISTRATION",
#         "EIGEN_NEON_GEBP_NR=4",
#     ] + select({
#         "//tensorflow:chromiumos": ["IS_MOBILE_PLATFORM"],
#         "//tensorflow:fuchsia": ["IS_MOBILE_PLATFORM"],
#         "//conditions:default": [],
#     }) + tf_defines_nortti_if_lite_protos() + select({
#         "//tensorflow:android": ["TF_USE_SNAPPY"],
#         "//conditions:default": [],
#     }),
#     features = ["-layering_check"],
#     tags = [
#         "manual",
#         "notap",
#     ],
#     visibility = ["//visibility:public"],
#     deps = [
#         "@com_google_absl//absl/algorithm:container",
#         "@com_google_absl//absl/base",
#         "@com_google_absl//absl/meta:type_traits",
#         "@com_google_absl//absl/strings",
#         "@com_google_absl//absl/types:optional",
#         "//tensorflow/core/util:managed_stack_trace",
#         "//tensorflow/core/platform:resource",
#         "//tensorflow/tsl/framework/fixedpoint",
#         "//tensorflow/core/util:stats_calculator_portable",
#         "//tensorflow/core:mobile_additional_lib_deps",
#     ] + tf_portable_proto_lib() + tf_portable_deps_no_runtime(),
#     alwayslink = 1,
# )
# copybara:uncomment_end

cc_library(
    name = "mobile_additional_lib_deps",
    deps = tf_additional_lib_deps() + [
        ":platform_base",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "portable_tensorflow_lib",
    srcs = if_mobile([":portable_op_registrations_and_gradients"]),
    copts = tf_copts() + tf_opts_nortti_if_lite_protos() + if_ios(["-Os"]),
    # Using private headers from other libraries violates layering check.
    features = tf_features_nomodules_if_mobile() + ["-layering_check"],
    tags = [
        "manual",
        "notap",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":portable_tensorflow_lib_lite",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/kernels:portable_tensorflow_kernels",
        "//third_party/eigen3",
        "@com_google_protobuf//:protobuf",
    ],
    alwayslink = 1,
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
        "//tensorflow/core/kernels:portable_tensorflow_kernels",
        "//tensorflow/tsl/platform/default/build_config:gtest",
        "//third_party/eigen3",
    ],
)

# -----------------------------------------------------------------------------
# Libraries with GPU facilities that are useful for writing kernels.
alias(
    name = "gpu_lib",
    actual =
        "//tensorflow/core/common_runtime/gpu:gpu_lib",
    visibility = ["//visibility:public"],
)

alias(
    name = "gpu_headers_lib",
    actual =
        "//tensorflow/core/common_runtime/gpu:gpu_headers_lib",
    visibility = ["//visibility:public"],
)

alias(
    name = "cuda",
    actual = "//tensorflow/tsl/platform:cuda",
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

# -----------------------------------------------------------------------------
# Internal targets
filegroup(
    name = "lib_internal_private_headers",
    srcs = [
        "//tensorflow/core/platform:legacy_lib_internal_headers",
        "//tensorflow/core/platform:lib_internal_private_hdrs",
        "//tensorflow/core/lib/core:legacy_lib_core_all_headers",
        "//tensorflow/tsl/lib/core:legacy_lib_core_all_headers",
        "//tensorflow/core/lib/gtl:legacy_lib_gtl_all_headers",
        "//tensorflow/core/lib/histogram:legacy_lib_histogram_all_headers",
        "//tensorflow/core/lib/hash:legacy_lib_hash_all_headers",
        "//tensorflow/core/lib/monitoring:legacy_lib_monitoring_all_headers",
        "//tensorflow/core/lib/io:legacy_lib_io_all_headers",
        "//tensorflow/core/lib/random:legacy_lib_random_all_headers",
        "//tensorflow/core/lib/strings:legacy_lib_strings_all_headers",
        "//tensorflow/core/lib/math:math_util.h",
        "//tensorflow/tsl/lib/io:legacy_lib_io_all_headers",
        "//tensorflow/tsl/lib/math:math_util.h",
        "//tensorflow/tsl/platform:legacy_lib_internal_headers",
        "//tensorflow/tsl/platform:lib_internal_private_hdrs",
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
        "//tensorflow/compiler/xla/stream_executor:device_mem_allocator_headers",
        "//tensorflow/core/lib/core:legacy_lib_internal_core_headers",
        "//tensorflow/core/lib/gtl:legacy_lib_internal_public_gtl_headers",
        "//tensorflow/core/lib/hash:legacy_lib_internal_public_headers",
        "//tensorflow/core/lib/io:legacy_lib_internal_public_headers",
        "//tensorflow/core/lib/llvm_rtti:legacy_lib_core_all_headers",
        "//tensorflow/core/lib/random:legacy_lib_internal_public_random_headers",
        "//tensorflow/core/lib/strings:legacy_lib_internal_public_string_headers",
        "//tensorflow/core/lib/wav:wav_io.h",
        "//tensorflow/core/platform:legacy_platform_lib_hdrs",
        "//tensorflow/core/platform:lib_internal_public_hdrs",
        "//tensorflow/core/util:lib_internal_public_hdrs",
        "//tensorflow/tsl/lib/io:legacy_lib_internal_public_headers",
        "//tensorflow/tsl/platform:lib_internal_public_hdrs",
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
        "@com_google_absl//absl/status",
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
    visibility = ["//tensorflow:internal"],
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
        "//tensorflow/core/lib/core:arena",
        "//tensorflow/core/lib/core:bitmap",
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
        "//tensorflow/core/lib/io:snappy_compression_options",
        "//tensorflow/core/lib/io:snappy_inputbuffer",
        "//tensorflow/core/lib/io:snappy_inputstream",
        "//tensorflow/core/lib/io:snappy_outputbuffer",
        "//tensorflow/core/lib/io:table",
        "//tensorflow/core/lib/io:table_options",
        "//tensorflow/core/lib/io:zlib_compression_options",
        "//tensorflow/core/lib/io:zlib_inputstream",
        "//tensorflow/core/lib/io:zlib_outputbuffer",
        "//tensorflow/core/lib/math:math_util",
        "//tensorflow/tsl/lib/math:math_util",
        "//tensorflow/core/lib/wav:wav_io",
        "//tensorflow/core/lib/monitoring:collected_metrics",
        "//tensorflow/core/lib/monitoring:collection_registry",
        "//tensorflow/core/lib/monitoring:counter",
        "//tensorflow/core/lib/monitoring:gauge",
        "//tensorflow/core/lib/monitoring:metric_def",
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
        "//tensorflow/core/platform:bfloat16",
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
        "//tensorflow/core/platform:env_impl",
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
        "//tensorflow/core/platform:profile_utils_cpu_utils",
        "//tensorflow/core/platform:protobuf_internal",
        "//tensorflow/core/platform:regexp",
        "//tensorflow/core/platform:resource",
        "//tensorflow/core/platform:scanner",
        "//tensorflow/core/platform:setround",
        "//tensorflow/core/platform:stacktrace",
        "//tensorflow/core/platform:stacktrace_handler",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:statusor",
        "//tensorflow/core/platform:strcat",
        "//tensorflow/core/platform:stringpiece",
        "//tensorflow/core/platform:stringprintf",
        "//tensorflow/core/platform:strong_hash",
        "//tensorflow/core/platform:str_util",
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
    ] + select({
        "//tensorflow:fuchsia": [],
        "//conditions:default": ["//tensorflow/core/platform:subprocess"],
    }) +
    # The TF proto implementations that we will statically link here.
    [
        "//tensorflow/compiler/mlir/quantization/tensorflow:exported_model_proto_cc_impl",
        "//tensorflow/compiler/mlir/quantization/tensorflow:quantization_options_proto_cc_impl",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc_impl",
        "//tensorflow/core/protobuf:autotuning_proto_cc_impl",
        "//tensorflow/core/protobuf:conv_autotuning_proto_cc_impl",
        "//tensorflow/core:protos_all_cc_impl",
        "//tensorflow/tsl/protobuf:autotuning_proto_cc_impl",
        "//tensorflow/tsl/profiler/protobuf:profiler_options_proto_cc_impl",
        "//tensorflow/tsl/protobuf:protos_all_cc_impl",
        "//tensorflow/compiler/xla:xla_proto_cc_impl",
        "//tensorflow/compiler/xla:xla_data_proto_cc_impl",
        "//tensorflow/compiler/xla/service:hlo_proto_cc_impl",
        "//tensorflow/compiler/xla/service/gpu:backend_configs_cc_impl",
    ] + tf_protos_grappler_impl() + tf_monitoring_framework_deps(),
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

alias(
    name = "gif_internal",
    actual = "//tensorflow/core/lib/gif:gif_internal",
)

alias(
    name = "jpeg_internal",
    actual = "//tensorflow/core/lib/jpeg:jpeg_internal",
)

cc_library(
    name = "tflite_portable_logging",
    hdrs = [
        "//tensorflow/core/platform:tflite_portable_logging_hdrs",
        "//tensorflow/tsl/platform:tflite_portable_logging_hdrs",
        "//tensorflow/tsl/platform/default:integral_types.h",
        "//tensorflow/tsl/platform/default:logging.h",
    ],
    compatible_with = get_compatible_with_portable(),
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

alias(
    name = "portable_jpeg_internal",
    actual = "//tensorflow/core/lib/jpeg:portable_jpeg_internal",
)

alias(
    name = "portable_gif_internal",
    actual = "//tensorflow/core/lib/gif:portable_gif_internal",
)

alias(
    name = "error_codes_proto_impl",
    actual = "//tensorflow/core/protobuf:error_codes_proto_impl",
)

alias(
    name = "error_codes_proto_impl_cc",
    actual = "//tensorflow/core/protobuf:error_codes_proto_impl_cc",
)

# copybara:uncomment_begin(google-only)
# alias(
#     name = "error_codes_proto",
#     actual = "//tensorflow/core/lib/core:error_codes_proto",
# )
# copybara:uncomment_end

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
        "//tensorflow/core/public:session_options.h",
        "//tensorflow/core/public:version.h",
        "//tensorflow/core/util:framework_internal_private_hdrs",
        "//tensorflow/core/util:memmapped_file_system_hdrs",
        "//tensorflow/core/util/sparse:framework_internal_private_headers_group",
        "//tensorflow/tsl/framework:framework_internal_private_hdrs",
    ],
    visibility = ["//visibility:private"],
)

filegroup(
    name = "framework_internal_public_headers",
    srcs = [
        "//tensorflow/core/framework:model.h",
        "//tensorflow/core/framework:op_segment.h",
        "//tensorflow/core/framework:ref_var.h",
        "//tensorflow/core/framework:rendezvous.h",  # only needed for tests
        "//tensorflow/core/framework:resource_var.h",
        "//tensorflow/core/framework:run_handler.h",
        "//tensorflow/core/framework:run_handler_util.h",
        "//tensorflow/core/framework:shared_ptr_variant.h",
        "//tensorflow/core/framework:tensor_reference.h",
        "//tensorflow/core/framework:tracking_allocator.h",  # only needed for tests
        "//tensorflow/core/framework:variant.h",
        "//tensorflow/core/util:framework_internal_public_hdrs",
        "//tensorflow/tsl/framework:tracking_allocator.h",  # only needed for tests
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
            "//tensorflow/core/util:managed_stack_trace",
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
        "//tensorflow/core/framework:bounds_check",
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
    ],
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
        "//tensorflow/core/framework:bounds_check",
        "//tensorflow/core/framework:common_shape_fns",
        "//tensorflow/core/framework:kernel_shape_util",
        "//tensorflow/core/framework:node_def_util",
        "//tensorflow/core/framework:node_properties",
        "//tensorflow/core/framework:numeric_types",
        "//tensorflow/core/framework:op",
        "//tensorflow/core/framework:op_def_builder",
        "//tensorflow/core/framework:op_def_util",
        "//tensorflow/core/framework:resource_handle",
        "//tensorflow/core/framework/registration",
        "//tensorflow/core/framework:shape_inference",
        "//tensorflow/core/framework:tensor",
        "//tensorflow/core/framework:tensor_shape",
        "//tensorflow/core/platform:env_impl",
        "//tensorflow/core/platform:fingerprint",
        "//tensorflow/core/platform/default/build_config:platformlib",
        "//tensorflow/tsl/platform:stringpiece",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/core/profiler/lib:annotated_traceme",
        "//tensorflow/core/profiler/lib:scoped_memory_debug_annotation",
        "//tensorflow/core/profiler/lib:traceme",
        "//tensorflow/core/util:determinism",
        "//tensorflow/core/util:managed_stack_trace",
        "//tensorflow/core/util:einsum_op_util",
        "//tensorflow/core/util:padding",
        "//tensorflow/core/util:port",
        "//tensorflow/core/util:stats_calculator_portable",
        "//tensorflow/core/util:tensor_format",
        "//tensorflow/compiler/jit:common",
        "//tensorflow/tsl/framework:cancellation",
        "//tensorflow/tsl/util:command_line_flags",
        "//tensorflow/tsl/util:device_name_utils",
        "@com_google_absl//absl/container:btree",
    ] + if_cuda([
        "@local_config_cuda//cuda:cudnn_header",
    ]) + if_static(
        extra_deps = ["@com_google_protobuf//:protobuf"],
        otherwise = ["@com_google_protobuf//:protobuf_headers"],
    ),
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
        "//tensorflow/core/platform:stream_executor",
    ],
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
        "//tensorflow/core/common_runtime:stats_publisher_interface",
        "//tensorflow/core/framework:bounds_check",
        "//tensorflow/core/kernels:required",
    ]),
    alwayslink = 1,
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
    name = "direct_session_internal",
    actual =
        "//tensorflow/core/common_runtime:direct_session_internal",
)

alias(
    name = "example_parser_configuration",
    actual = "//tensorflow/core/example:example_parser_configuration",
    visibility = ["//visibility:public"],
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
        "//tensorflow/tsl/lib/io:legacy_lib_test_internal_headers",
    ],
    deps = [
        ":lib",
        ":lib_internal",
    ],
)

# Main program for tests
alias(
    name = "test_main",
    actual = "//tensorflow/tsl/platform:test_main",
    visibility = ["//tensorflow:internal"],
)

tf_cc_tests(
    name = "low_level_library_tests",
    size = "small",
    srcs = [
        "//tensorflow/core/lib/core:legacy_lib_core_all_tests",
        "//tensorflow/core/lib/gtl:legacy_lib_gtl_tests",
        "//tensorflow/core/lib/monitoring:collection_registry_test.cc",
        "//tensorflow/core/lib/monitoring:counter_test.cc",
        "//tensorflow/core/lib/monitoring:gauge_test.cc",
        "//tensorflow/core/lib/monitoring:metric_def_test.cc",
        "//tensorflow/core/lib/monitoring:percentile_sampler_test.cc",
        "//tensorflow/core/lib/monitoring:sampler_test.cc",
        "//tensorflow/core/lib/strings:legacy_low_level_library_tests",
        "//tensorflow/core/lib/wav:wav_io_test.cc",
        "//tensorflow/tsl/lib/core:legacy_lib_core_all_tests",
    ],
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

tf_cc_tests(
    name = "cell_reader_test",
    size = "small",
    srcs = [
        "//tensorflow/core/lib/monitoring:cell_reader_test.cc",
        "//tensorflow/core/lib/monitoring:test_utils_test.cc",
    ],
    deps = [
        ":test",
        ":test_main",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/lib/monitoring:cell_reader",
        "//tensorflow/core/lib/monitoring:counter",
        "//tensorflow/core/lib/monitoring:gauge",
        "//tensorflow/core/lib/monitoring:percentile_sampler",
        "//tensorflow/core/lib/monitoring:sampler",
        "//tensorflow/core/lib/monitoring:test_utils",
        "//tensorflow/core/lib/monitoring:types",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:protobuf",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:status_matchers",
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
        "//tensorflow/core/kernels/mkl:mkl_aggregate_ops",
        "//tensorflow/core/kernels/mkl:mkl_batch_matmul_op",
        "//tensorflow/core/kernels/mkl:mkl_einsum_op",
        "//tensorflow/core/kernels/mkl:mkl_concat_op",
        "//tensorflow/core/kernels/mkl:mkl_conv_op",
        "//tensorflow/core/kernels/mkl:mkl_cwise_ops_common",
        "//tensorflow/core/kernels/mkl:mkl_dequantize_op",
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

# Test data
filegroup(
    name = "image_testdata",
    srcs = [
        # PNG data
        "//tensorflow/core/lib/png:testdata",
        "//tensorflow/core/lib/ssim:testdata",
        "//tensorflow/core/lib/psnr:testdata",
        # JPEG data
        "//tensorflow/core/lib/jpeg/testdata",
        # GIF data
        "//tensorflow/core/lib/gif/testdata:gif_testdata",
        # BMP data
        "//tensorflow/core/lib/bmp:bmp_testdata",
    ],
    visibility = ["//visibility:public"],
)

transitive_hdrs(
    name = "headers",
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        ":core",
        ":framework",
        ":lib",
        ":protos_all_cc",
        "//tensorflow/core/platform:platform_strings",
        "//tensorflow/core/platform:stream_executor",
    ],
)

# copybara:uncomment_begin(google-only)
# py_proto_library(
#     name = "protos_all_py_pb2",
#     has_services = 0,
#     api_version = 2,
#     visibility = ["//visibility:public"],
#     deps = [":protos_all"],
# )
#
# alias(
#     name = "worker_service_cc_grpc_proto",
#     actual = "//tensorflow/core/protobuf:worker_service_cc_grpc_proto",
#     visibility = ["//tensorflow:internal"],
# )
#
# alias(
#     name = "master_service_cc_grpc_proto",
#     actual = "//tensorflow/core/protobuf:master_service_cc_grpc_proto",
#     visibility = ["//tensorflow:internal"],
# )
#
# go_library(
#     name = "protos_all_go_proto",
#     srcs = ["protos_all_go_proto.go"],
#     visibility = ["//visibility:public"],
#     deps = [
#         "//tensorflow/core/example:example_parser_configuration_go_proto",
#         "//tensorflow/core/example:example_protos_go_proto",
#         "//tensorflow/core/protobuf:for_core_protos_go_proto",
#         "//tensorflow/core/util:event_go_proto",
#         "//tensorflow/core/util:memmapped_file_system_go_proto",
#         "//tensorflow/core/util:saved_tensor_slice_go_proto",
#     ],
# )
# copybara:uncomment_end

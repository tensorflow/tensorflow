# Description:
# TensorFlow is a computational framework, primarily for use in machine
# learning applications.
#
# Public targets:
#
# ":protos_all_cc" - exports all core TensorFlow protos
#     ":protos_all_py" - py_proto_library version (Google-internal)
# ":lib" - exports the public non-test headers for:
#     platform/: Platform-specific code and external dependencies
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

package(default_visibility = [
    "//tensorflow:internal",
    "//tensorflow_models:__subpackages__",
])

licenses(["notice"])  # Apache 2.0

# Export the BUILD file so automated tooling can check licenses
exports_files(["BUILD"])

load(
    "//tensorflow:tensorflow.bzl",
    "cc_header_only_library",
    "if_android",
    "if_ios",
    "if_linux_x86_64",
    "if_mobile",
    "if_not_windows",
    "if_windows",
    "tf_cc_test",
    "tf_cc_tests",
    "tf_copts",
    "tf_cuda_library",
    "tf_features_nomodules_if_android",
    "tf_gen_op_libs",
    "tf_generate_proto_text_sources",
    "tf_genrule_cmd_append_to_srcs",
    "tf_opts_nortti_if_android",
    "transitive_hdrs",
)
load("//tensorflow:tensorflow.bzl", "tf_cc_test_mkl")
load("//tensorflow:tensorflow.bzl", "tf_cc_test_gpu")
load("//tensorflow:tensorflow.bzl", "tf_cc_tests_gpu")
load("//tensorflow:tensorflow.bzl", "tf_cuda_cc_test")
load("//tensorflow:tensorflow.bzl", "tf_version_info_genrule")
load("//tensorflow:tensorflow.bzl", "if_nccl")
load("//tensorflow:tensorflow.bzl", "tensorflow_opensource_extra_deps")
load("//tensorflow:tensorflow.bzl", "tf_cuda_only_cc_test")

# For platform specific build config
load(
    "//tensorflow/core:platform/default/build_config.bzl",
    "tf_additional_all_protos",
    "tf_additional_cloud_kernel_deps",
    "tf_additional_cloud_op_deps",
    "tf_additional_core_deps",
    "tf_additional_cupti_wrapper_deps",
    "tf_additional_device_tracer_cuda_deps",
    "tf_additional_device_tracer_deps",
    "tf_additional_device_tracer_srcs",
    "tf_additional_device_tracer_test_flags",
    "tf_additional_gdr_lib_defines",
    "tf_additional_human_readable_json_deps",
    "tf_additional_lib_defines",
    "tf_additional_lib_deps",
    "tf_additional_lib_hdrs",
    "tf_additional_lib_srcs",
    "tf_additional_libdevice_data",
    "tf_additional_libdevice_deps",
    "tf_additional_libdevice_srcs",
    "tf_additional_minimal_lib_srcs",
    "tf_additional_mpi_lib_defines",
    "tf_additional_proto_compiler_hdrs",
    "tf_additional_proto_hdrs",
    "tf_additional_proto_srcs",
    "tf_additional_test_deps",
    "tf_additional_test_srcs",
    "tf_additional_verbs_lib_defines",
    "tf_jspb_proto_library",
    "tf_kernel_tests_linkstatic",
    "tf_lib_proto_compiler_deps",
    "tf_lib_proto_parsing_deps",
    "tf_platform_hdrs",
    "tf_platform_srcs",
    "tf_proto_library",
    "tf_proto_library_cc",
    "tf_protos_all",
    "tf_protos_all_impl",
    "tf_protos_grappler",
    "tf_protos_grappler_impl",
    "tf_pyclif_proto_library",
)
load(
    "//tensorflow/core:platform/default/build_config_root.bzl",
    "if_dynamic_kernels",
    "if_static",
    "tf_cuda_tests_tags",
)
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda_is_configured")
load("@io_bazel_rules_closure//closure:defs.bzl", "closure_proto_library")
load(
    "//third_party/mkl:build_defs.bzl",
    "if_mkl",
    "mkl_deps",
)

exports_files(["ops/ops.pbtxt"])

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
    "protobuf/config.proto",
    "protobuf/cluster.proto",
    "protobuf/debug.proto",
    "protobuf/device_properties.proto",
    "protobuf/queue_runner.proto",
    "protobuf/rewriter_config.proto",
    "protobuf/tensor_bundle.proto",
    "protobuf/saver.proto",
    "protobuf/verifier_config.proto",
    "util/event.proto",
    "util/memmapped_file_system.proto",
    "util/saved_tensor_slice.proto",
]

ERROR_CODES_PROTO_SRCS = [
    "lib/core/error_codes.proto",
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
    "protobuf/checkpointable_object_graph.proto",
    "protobuf/control_flow.proto",
    # TODO(ebrevdo): Re-enable once CriticalSection is in core.
    # "protobuf/critical_section.proto",
    "protobuf/meta_graph.proto",
    "protobuf/named_tensor.proto",
    "protobuf/saved_model.proto",
    "protobuf/tensorflow_server.proto",
    "protobuf/transport_options.proto",
    "util/test_log.proto",
]

tf_proto_library(
    name = "protos_all",
    srcs = [],
    cc_api_version = 2,
    default_header = True,
    protodeps = [
        ":protos_all_proto",
        ":error_codes_proto",
    ],
    visibility = ["//visibility:public"],
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

exports_files([
    "framework/types.proto",
])

tf_proto_library(
    name = "protos_test",
    srcs = ["util/example_proto_fast_parsing_test.proto"],
    cc_api_version = 2,
    protodeps = tf_additional_all_protos(),
    visibility = ["//visibility:public"],
)

# Minimal lib to detect platform
cc_library(
    name = "lib_platform",
    hdrs = [
        "platform/platform.h",
    ],
)

filegroup(
    name = "platform_base_hdrs",
    srcs = [
        "platform/byte_order.h",
        "platform/cord.h",
        "platform/env_time.h",
        "platform/logging.h",
        "platform/macros.h",
        "platform/platform_strings.h",
        "platform/types.h",
    ],
    visibility = ["//visibility:private"],
)

cc_library(
    name = "platform_base",
    srcs = tf_platform_hdrs([
        "integral_types.h",
        "logging.h",
    ]) + tf_platform_srcs([
        "logging.cc",
        "env_time.cc",
    ]) + [
        "platform/env_time.cc",
    ],
    hdrs = [":platform_base_hdrs"],
    copts = tf_copts(),
    tags = ["avoid_dep"],
    visibility = ["//tensorflow/core:__subpackages__"],
    deps = [
        ":lib_platform",
        "//tensorflow/core/platform/default/build_config:base",
    ],
)

cc_library(
    name = "framework_bounds_check",
    hdrs = ["framework/bounds_check.h"],
    visibility = ["//tensorflow/core/kernels:friends"],
    deps = [
        "//tensorflow/core:platform_base",
        "//third_party/eigen3",
    ],
)

filegroup(
    name = "platform_port_hdrs",
    srcs = [
        "platform/cpu_info.h",
        "platform/dynamic_annotations.h",
        "platform/init_main.h",
        "platform/mem.h",
        "platform/mutex.h",
        "platform/numa.h",
        "platform/thread_annotations.h",
    ],
    visibility = ["//visibility:private"],
)

# Headers that are not exported as part of ":lib".
filegroup(
    name = "platform_port_internal_hdrs",
    srcs = [
        "platform/demangle.h",
        "platform/host_info.h",
        "platform/snappy.h",
    ],
    visibility = ["//visibility:private"],
)

cc_library(
    name = "platform_port",
    srcs = tf_platform_hdrs([
        "cpu_info.h",
        "dynamic_annotations.h",
        "thread_annotations.h",
        "mutex.h",
    ]) + tf_platform_srcs([
        "port.cc",
    ]) + [
        "platform/cpu_info.cc",
    ],
    hdrs = [
        ":platform_port_hdrs",
        ":platform_port_internal_hdrs",
    ],
    copts = tf_copts(),
    visibility = ["//tensorflow/core:__subpackages__"],
    deps = [
        ":lib_platform",
        ":platform_base",
        "//tensorflow/core/platform/default/build_config:port",
        "@com_google_absl//absl/base",
        "@snappy",
    ],
)

filegroup(
    name = "platform_protobuf_hdrs",
    srcs = [
        "platform/protobuf.h",
    ],
    visibility = ["//visibility:private"],
)

# Headers that are not exported as part of ":lib".
filegroup(
    name = "platform_protobuf_internal_hdrs",
    srcs = [
        "platform/protobuf_internal.h",
    ],
    visibility = ["//visibility:private"],
)

cc_library(
    name = "platform_protobuf",
    srcs = tf_platform_hdrs([
        "protobuf.h",
    ]) + tf_platform_srcs([
        "protobuf.cc",
    ]) + [
        "platform/protobuf_util.cc",
        "lib/core/status.h",
    ],
    hdrs = [
        ":platform_protobuf_hdrs",
        ":platform_protobuf_internal_hdrs",
    ],
    copts = tf_copts(),
    visibility = ["//tensorflow/core:__subpackages__"],
    deps = [
        ":lib_platform",
        ":platform_base",
        ":platform_port",
        "//tensorflow/core/platform/default/build_config:protobuf",
        "@protobuf_archive//:protobuf",
    ],
)

cc_library(
    name = "human_readable_json",
    srcs = tf_platform_srcs(["human_readable_json.cc"]),
    hdrs = ["platform/human_readable_json.h"],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":lib",
        ":lib_internal",
    ] + tf_additional_human_readable_json_deps(),
)

cc_library(
    name = "logger",
    srcs = ["platform/logger.cc"],
    hdrs = ["platform/logger.h"],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":lib_proto_parsing",
        "@protobuf_archive//:protobuf",
    ],
)

filegroup(
    name = "platform_env_hdrs",
    srcs = [
        "platform/env.h",
        "platform/file_statistics.h",
        "platform/file_system.h",
    ],
    visibility = ["//visibility:private"],
)

# Headers that are not exported as part of ":lib".
filegroup(
    name = "platform_env_internal_hdrs",
    srcs = [
        "platform/load_library.h",
    ],
    visibility = ["//visibility:private"],
)

cc_library(
    name = "platform_env",
    srcs = tf_platform_srcs([
        "env.cc",
        "load_library.cc",
    ]) + tf_platform_hdrs([
        "wide_char.h",
    ]) + [
        "platform/env.cc",
        "platform/file_system.cc",
    ],
    hdrs = [
        ":platform_env_hdrs",
        ":platform_env_internal_hdrs",
    ],
    copts = tf_copts(),
    visibility = [
        "//tensorflow/c:__subpackages__",
        "//tensorflow/core:__subpackages__",
    ],
    deps = [
        ":error_codes_proto_cc",
        ":lib",
        ":lib_internal",
        ":lib_platform",
        ":platform_base",
        ":platform_port",
        ":platform_protobuf",
        "//tensorflow/core/platform/default/build_config:env",
        "//tensorflow/core/platform/default/build_config:port",
    ],
)

filegroup(
    name = "platform_file_system_hdrs",
    srcs = [
        "platform/file_system_helper.h",
        "platform/null_file_system.h",
    ],
    visibility = ["//visibility:private"],
)

cc_library(
    name = "platform_file_system",
    srcs = tf_platform_srcs([
    ]) + tf_platform_hdrs([
        "windows_file_system.h",
    ]) + [
        "platform/file_system_helper.cc",
    ],
    hdrs = [
        ":platform_file_system_hdrs",
    ],
    copts = tf_copts(),
    visibility = ["//tensorflow/core:__subpackages__"],
    deps = [
        ":lib",
        ":lib_platform",
        ":platform_env",
    ],
)

cc_library(
    name = "platform_strings",
    srcs = tf_platform_srcs([
        "platform/platform_strings.cc",
        "platform/platform_strings_computed.h",
    ]),
    hdrs = [
        "platform/platform_strings.h",
    ],
    visibility = ["//tensorflow/core:__subpackages__"],
    deps = [":lib"],
)

filegroup(
    name = "platform_other_hdrs",
    srcs = [
        "platform/abi.h",
        "platform/context.h",
        "platform/cpu_feature_guard.h",
        "platform/error.h",
        "platform/fingerprint.h",
        "platform/net.h",
        "platform/notification.h",
        "platform/prefetch.h",
        "platform/profile_utils/android_armv7a_cpu_utils_helper.h",
        "platform/profile_utils/clock_cycle_profiler.h",
        "platform/profile_utils/cpu_utils.h",
        "platform/profile_utils/i_cpu_utils_helper.h",
        "platform/stacktrace.h",
        "platform/stacktrace_handler.h",
        "platform/strong_hash.h",
        "platform/subprocess.h",
    ],
    visibility = ["//visibility:private"],
)

# Headers that are not exported as part of ":lib".
filegroup(
    name = "platform_other_internal_hdrs",
    srcs = [
        "platform/denormal.h",
        "platform/setround.h",
        "platform/tracing.h",
    ],
    visibility = ["//visibility:private"],
)

cc_library(
    name = "platform_other",
    srcs = tf_platform_srcs([
        "subprocess.cc",
        "net.cc",
        "tracing.cc",
    ]) + tf_platform_hdrs([
        "tracing.h",
        "error.h",
        "context.h",
        "fingerprint.h",
        "notification.h",
        "stacktrace.h",
        "strong_hash.h",
        "subprocess.h",
        "tracing_impl.h",
    ]) + [
        "platform/cpu_feature_guard.cc",
        "platform/setround.cc",
        "platform/tracing.cc",
        "platform/denormal.cc",
        "platform/profile_utils/android_armv7a_cpu_utils_helper.cc",
        "platform/profile_utils/clock_cycle_profiler.cc",
        "platform/profile_utils/cpu_utils.cc",
    ],
    hdrs = [
        ":platform_other_hdrs",
        ":platform_other_internal_hdrs",
    ],
    copts = tf_copts(),
    visibility = ["//tensorflow/core:__subpackages__"],
    deps = [
        ":lib",
        ":lib_platform",
        ":platform_base",
        ":platform_env",
        ":platform_port",
        ":platform_protobuf",
        "//tensorflow/core/platform/default/build_config:other",
        "//tensorflow/core/platform/default/build_config:platformlib",
        "//tensorflow/core/platform/default/build_config:port",
    ],
)

# Minimal lib so that tools used for mobile compilation
# don't have to depend on lib/platformlib.
cc_library(
    name = "lib_proto_parsing",
    srcs = glob(tf_additional_proto_srcs()),
    hdrs = [
        "lib/core/errors.h",
        "lib/core/status.h",
        "lib/core/stringpiece.h",
        "lib/strings/numbers.h",
        "lib/strings/strcat.h",
        "platform/init_main.h",
        "platform/logging.h",
        "platform/macros.h",
        "platform/platform.h",
        "platform/protobuf.h",
        "platform/types.h",
        "platform/windows/cpu_info.h",
        "lib/bfloat16/bfloat16.h",
    ] + tf_additional_proto_hdrs(),
    copts = tf_copts(),
    deps = tf_lib_proto_parsing_deps() + [
        ":platform_base",
        "@com_google_absl//absl/strings",
        "@double_conversion//:double-conversion",
    ],
)

cc_library(
    name = "lib_proto_compiler",
    hdrs = [
        "platform/protobuf_compiler.h",
    ] + tf_additional_proto_compiler_hdrs(),
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
        "lib/bfloat16/bfloat16.h",
        "lib/core/arena.h",
        "lib/core/bitmap.h",
        "lib/core/bits.h",
        "lib/core/coding.h",
        "lib/core/errors.h",
        "lib/core/notification.h",
        "lib/core/raw_coding.h",
        "lib/core/status.h",
        "lib/core/stringpiece.h",
        "lib/core/threadpool.h",
        "lib/gtl/array_slice.h",
        "lib/gtl/cleanup.h",
        "lib/gtl/compactptrset.h",
        "lib/gtl/flatmap.h",
        "lib/gtl/flatset.h",
        "lib/gtl/inlined_vector.h",
        "lib/gtl/optional.h",
        "lib/gtl/priority_queue_util.h",
        "lib/hash/crc32c.h",
        "lib/hash/hash.h",
        "lib/histogram/histogram.h",
        "lib/io/buffered_inputstream.h",
        "lib/io/compression.h",
        "lib/io/inputstream_interface.h",
        "lib/io/path.h",
        "lib/io/proto_encode_helper.h",
        "lib/io/random_inputstream.h",
        "lib/io/record_reader.h",
        "lib/io/record_writer.h",
        "lib/io/table.h",
        "lib/io/table_builder.h",
        "lib/io/table_options.h",
        "lib/math/math_util.h",
        "lib/monitoring/collected_metrics.h",
        "lib/monitoring/collection_registry.h",
        "lib/monitoring/counter.h",
        "lib/monitoring/gauge.h",
        "lib/monitoring/metric_def.h",
        "lib/monitoring/sampler.h",
        "lib/random/distribution_sampler.h",
        "lib/random/philox_random.h",
        "lib/random/random_distributions.h",
        "lib/random/simple_philox.h",
        "lib/strings/numbers.h",
        "lib/strings/proto_serialization.h",
        "lib/strings/str_util.h",
        "lib/strings/strcat.h",
        "lib/strings/stringprintf.h",
        ":platform_base_hdrs",
        ":platform_env_hdrs",
        ":platform_file_system_hdrs",
        ":platform_other_hdrs",
        ":platform_port_hdrs",
        ":platform_protobuf_hdrs",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":lib_internal",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
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
)

cc_library(
    name = "abi",
    srcs = ["platform/abi.cc"],
    hdrs = ["platform/abi.h"],
    deps = [":platform_base"],
)

cc_library(
    name = "stacktrace",
    srcs = glob(["platform/*/stacktrace.h"]),
    hdrs = ["platform/stacktrace.h"],
    deps = [
        ":abi",
        ":lib_platform",
        "//tensorflow/core/platform/default/build_config:stacktrace",
    ],
)

cc_library(
    name = "stacktrace_handler",
    srcs = ["platform/stacktrace_handler.cc"],
    hdrs = ["platform/stacktrace_handler.h"],
    deps = [
        ":abi",
        ":lib_platform",
        ":stacktrace",
    ],
)

# Libraries that will eventually be moved into lib/core
# Note that stringpiece_test can't be place here yet, because we are
# required to use tf_cc_test, and that rule will change / into _
cc_library(
    name = "core_stringpiece",
    hdrs = ["lib/core/stringpiece.h"],
    copts = tf_copts(),
    deps = [
        ":platform_base",
        "@com_google_absl//absl/strings",
    ],
)

# Test support library needed for all tests
# This is currently public, but may be made internal in the
# future.  Try to avoid depending on it.
cc_library(
    name = "test",
    testonly = 1,
    srcs = [
        "platform/test.cc",
        "util/reporter.cc",
    ] + tf_additional_test_srcs(),
    hdrs = [
        "lib/core/status_test_util.h",
        "platform/test.h",
        "platform/test_benchmark.h",
        "util/reporter.h",
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
    ] + tf_additional_test_deps(),
)

# Testing libraries - lite versions that don't depend on all of "lib" or
# "lib_internal". Instead, they only need a much smaller set of support
# libraries such as ":platform_base" and ":core_stringpiece".
cc_library(
    name = "test_lite",
    testonly = 1,
    srcs = [
        "platform/test.cc",
    ],
    hdrs = [
        "platform/test.h",
        "platform/test_benchmark.h",
    ],
    copts = tf_copts(),
    deps = [
        ":lib_platform",
        ":platform_base",
        "//tensorflow/core/platform/default/build_config:gtest",
    ],
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
        "framework/tracking_allocator.h",
        "framework/type_index.h",
        "framework/type_traits.h",
        "framework/types.h",
        "public/version.h",
        "util/activation_mode.h",
        "util/batch_util.h",
        "util/bcast.h",
        "util/cuda_kernel_helper.h",
        "util/device_name_utils.h",
        "util/dump_graph.h",
        "util/events_writer.h",
        "util/example_proto_fast_parsing.h",
        "util/example_proto_helper.h",
        "util/guarded_philox_random.h",
        "util/mirror_pad_mode.h",
        "util/padding.h",
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
    deps = [":framework_internal"],
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
    srcs = tf_additional_minimal_lib_srcs(),
    hdrs = [
        "framework/numeric_types.h",
        "framework/tensor_types.h",
        "framework/type_traits.h",
        "lib/bfloat16/bfloat16.h",
        "platform/byte_order.h",
        "platform/default/dynamic_annotations.h",
        "platform/default/integral_types.h",
        "platform/default/logging.h",
        "platform/default/mutex.h",
        "platform/default/protobuf.h",
        "platform/default/thread_annotations.h",
        "platform/dynamic_annotations.h",
        "platform/macros.h",
        "platform/mutex.h",
        "platform/platform.h",
        "platform/prefetch.h",
        "platform/thread_annotations.h",
        "platform/types.h",
        "platform/cpu_info.h",
    ] + if_windows(["platform/windows/integral_types.h"]),
    visibility = ["//visibility:public"],
    deps =
        [
            "@nsync//:nsync_cpp",
        ] + [
            "//third_party/eigen3",
            "//tensorflow/core/platform/default/build_config:minimal",
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
        "rpc_ops",
        "scoped_allocator_ops",
        "sdca_ops",
        "set_ops",
        "script_ops",
        "sendrecv_ops",
        "sparse_ops",
        "spectral_ops",
        "state_ops",
        "stateless_random_ops",
        "summary_ops",
        "training_ops",
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
    deps = ["//tensorflow/core/kernels:debug_ops"],
)

tf_gen_op_libs(
    is_external = False,
    op_lib_names = [
        "resource_variable_ops",
    ],
    deps = [":lib"],
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
    deps = ["//tensorflow/core:framework"],
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
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:stream_executor",
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
        ":stateful_random_ops_op_lib",
        ":remote_fused_graph_ops_op_lib",
        ":resource_variable_ops_op_lib",
        ":rpc_ops_op_lib",
        ":scoped_allocator_ops_op_lib",
        ":script_ops_op_lib",
        ":sdca_ops_op_lib",
        ":sendrecv_ops_op_lib",
        ":set_ops_op_lib",
        ":sparse_ops_op_lib",
        ":summary_ops_op_lib",
        ":spectral_ops_op_lib",
        ":state_ops_op_lib",
        ":stateless_random_ops_op_lib",
        ":string_ops_op_lib",
        ":training_ops_op_lib",
        ":user_ops_op_lib",
        ":word2vec_ops",
    ] + if_mkl([":mkl_nn_ops_op_lib"]) + tf_additional_cloud_op_deps(),
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
    visibility = ["//visibility:private"],
    deps = [
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
        "//tensorflow/core/kernels:random_poisson_op",
        "//tensorflow/core/kernels:remote_fused_graph_ops",
        "//tensorflow/core/kernels:required",
        "//tensorflow/core/kernels:resource_variable_ops",
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
    ] + tf_additional_cloud_kernel_deps() + if_not_windows([
        "//tensorflow/core/kernels:fact_op",
        "//tensorflow/core/kernels:array_not_windows",
        "//tensorflow/core/kernels:math_not_windows",
        "//tensorflow/core/kernels:quantized_ops",
        "//tensorflow/core/kernels/neon:neon_depthwise_conv_op",
    ]) + if_mkl([
        "//tensorflow/core/kernels:mkl_concat_op",
        "//tensorflow/core/kernels:mkl_conv_op",
        "//tensorflow/core/kernels:mkl_cwise_ops_common",
        "//tensorflow/core/kernels:mkl_fused_batch_norm_op",
        "//tensorflow/core/kernels:mkl_identity_op",
        "//tensorflow/core/kernels:mkl_input_conversion_op",
        "//tensorflow/core/kernels:mkl_lrn_op",
        "//tensorflow/core/kernels:mkl_pooling_ops",
        "//tensorflow/core/kernels:mkl_relu_op",
        "//tensorflow/core/kernels:mkl_reshape_op",
        "//tensorflow/core/kernels:mkl_slice_op",
        "//tensorflow/core/kernels:mkl_softmax_op",
        "//tensorflow/core/kernels:mkl_transpose_op",
        "//tensorflow/core/kernels:mkl_tfconv_op",
        "//tensorflow/core/kernels:mkl_aggregate_ops",
    ]) + if_cuda([
        "//tensorflow/core/grappler/optimizers:gpu_swapping_kernels",
        "//tensorflow/core/grappler/optimizers:gpu_swapping_ops",
    ]) + if_nccl([
        "//tensorflow/core/kernels:nccl_kernels",
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
            "//tensorflow/core/kernels:cast_op",
            "//tensorflow/core/kernels:constant_op",
            "//tensorflow/core/kernels:random_ops",
        ],
    ),
)

cc_library(
    name = "testlib_ops",
    testonly = 1,
    srcs = ["common_runtime/testlib_ops.cc"],
    linkstatic = 1,  # Seems to be needed since alwayslink is broken in bazel
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
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
# Public Android targets

# Android-specific BUILD targets
load(
    "//tensorflow:tensorflow.bzl",
    "tf_android_core_proto_headers",
    "tf_android_core_proto_sources",
)

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
        ":protos_all_proto_text_srcs",
        ":error_codes_proto_text_srcs",
        "//tensorflow/core/platform/default/build_config:android_srcs",
    ] + glob(
        [
            "client/**/*.cc",
            "framework/**/*.h",
            "framework/**/*.cc",
            "lib/**/*.h",
            "lib/**/*.cc",
            "platform/**/*.h",
            "platform/**/*.cc",
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
            "util/events_writer.*",
            "util/stats_calculator.*",
            "util/reporter.*",
            "platform/**/cuda_libdevice_path.*",
            "platform/**/logger.cc",
            "platform/default/test_benchmark.*",
            "platform/cuda.h",
            "platform/google/**/*",
            "platform/hadoop/**/*",
            "platform/gif.h",
            "platform/jpeg.h",
            "platform/png.h",
            "platform/stream_executor.*",
            "platform/windows/**/*",
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
    visibility = ["//visibility:public"],
)

filegroup(
    name = "mobile_srcs_only_runtime",
    srcs = [
        "//tensorflow/core/kernels:android_srcs",
        "//tensorflow/core/util/ctc:android_srcs",
        "//tensorflow/core/util/tensor_bundle:android_srcs",
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
            "common_runtime/eager/*",
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
    srcs = if_android(["//tensorflow/core:android_srcs"]),
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
        "@double_conversion//:double-conversion",
        "@nsync//:nsync_cpp",
        "@protobuf_archive//:protobuf",
    ],
    alwayslink = 1,
)

cc_library(
    name = "android_tensorflow_lib_lite_nortti",
    srcs = if_android(["//tensorflow/core:android_srcs"]),
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
        "@double_conversion//:double-conversion",
        "@nsync//:nsync_cpp",
        "@protobuf_archive//:protobuf",
    ],
    alwayslink = 1,
)

cc_library(
    name = "mobile_additional_lib_deps",
    deps = tf_additional_lib_deps() + [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

# Native library support for iOS applications.
#
# bazel  build --config=ios_x86_64 \
# //third_party/tensorflow/core:ios_tensorflow_lib
cc_library(
    name = "ios_tensorflow_lib",
    srcs = if_ios([
        ":android_op_registrations_and_gradients",
        "//tensorflow/core/kernels:android_core_ops",
        "//tensorflow/core/kernels:android_extended_ops",
    ]),
    copts = tf_copts() + ["-Os"] + ["-std=c++11"],
    visibility = ["//visibility:public"],
    deps = [
        ":ios_tensorflow_lib_lite",
        ":protos_all_cc_impl",
        "//third_party/eigen3",
        "//third_party/fft2d:fft2d_headers",
        "@fft2d",
        "@gemmlowp",
        "@protobuf_archive//:protobuf",
    ],
    alwayslink = 1,
)

cc_library(
    name = "ios_tensorflow_lib_lite",
    srcs = if_ios(["//tensorflow/core:android_srcs"]),
    copts = tf_copts() + ["-Os"] + ["-std=c++11"],
    visibility = ["//visibility:public"],
    deps = [
        ":mobile_additional_lib_deps",
        ":protos_all_cc_impl",
        ":stats_calculator_portable",
        "//third_party/eigen3",
        "@double_conversion//:double-conversion",
        "@nsync//:nsync_cpp",
        "@protobuf_archive//:protobuf",
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
        "@protobuf_archive//:protobuf",
    ],
    alwayslink = 1,
)

filegroup(
    name = "android_op_registrations_and_gradients",
    srcs = glob(
        [
            "ops/**/*.cc",
            "ops/**/*.h",
        ],
        exclude = [
            "**/*test.cc",
            "**/*testutil*",
            "**/*testlib*",
            "**/*main.cc",
        ],
    ),
    visibility = ["//visibility:public"],
)

filegroup(
    name = "android_test_srcs",
    # TODO(andrewharp/nhua):
    # make more test-related sources portable e.g. "platform/test.cc",
    srcs = [
        ":framework/fake_input.cc",
        ":framework/fake_input.h",
        ":framework/shape_inference_testutil.cc",
        ":framework/shape_inference_testutil.h",
        ":framework/tensor_testutil.cc",
        ":framework/tensor_testutil.h",
        ":platform/test.cc",
        ":platform/test.h",
        ":util/reporter.cc",
        ":util/reporter.h",
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
        ":platform/test.h",
        ":util/reporter.cc",
        ":util/reporter.h",
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

# -----------------------------------------------------------------------------
# Internal targets

tf_proto_library_cc(
    name = "worker_proto",
    srcs = ["protobuf/worker.proto"],
    cc_api_version = 2,
    protodeps = tf_additional_all_protos() + [],
    visibility = [
        "//tensorflow:internal",
    ],
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
    cc_stubby_versions = ["2"],
    protodeps = tf_additional_all_protos(),
    visibility = [
        "//tensorflow:internal",
    ],
)

LIB_INTERNAL_PRIVATE_HEADERS = ["framework/resource_handle.h"] + glob(
    [
        "lib/**/*.h",
        "platform/*.h",
        "platform/profile_utils/**/*.h",
    ],
    exclude = [
        "**/*test*",
        "lib/gif/**/*",
        "lib/jpeg/**/*",
        "lib/png/**/*",
        "platform/gif.h",
        "platform/jpeg.h",
        "platform/png.h",
        "platform/**/cuda.h",
        "platform/**/stream_executor.h",
    ],
)

LIB_INTERNAL_PUBLIC_HEADERS = tf_additional_lib_hdrs() + [
    "lib/core/blocking_counter.h",
    "lib/core/refcount.h",
    "lib/gtl/edit_distance.h",
    "lib/gtl/int_type.h",
    "lib/gtl/iterator_range.h",
    "lib/gtl/manual_constructor.h",
    "lib/gtl/map_util.h",
    "lib/gtl/stl_util.h",
    "lib/gtl/top_n.h",
    "lib/hash/hash.h",
    "lib/io/inputbuffer.h",
    "lib/io/iterator.h",
    "lib/io/snappy/snappy_inputbuffer.h",
    "lib/io/snappy/snappy_outputbuffer.h",
    "lib/io/zlib_compression_options.h",
    "lib/io/zlib_inputstream.h",
    "lib/io/zlib_outputbuffer.h",
    "lib/monitoring/mobile_counter.h",
    "lib/monitoring/mobile_gauge.h",
    "lib/monitoring/mobile_sampler.h",
    "lib/png/png_io.h",
    "lib/random/random.h",
    "lib/random/random_distributions.h",
    "lib/random/weighted_picker.h",
    "lib/strings/base64.h",
    "lib/strings/ordered_code.h",
    "lib/strings/proto_text_util.h",
    "lib/strings/proto_serialization.h",
    "lib/strings/scanner.h",
    "lib/wav/wav_io.h",
    "platform/demangle.h",
    "platform/denormal.h",
    "platform/host_info.h",
    "platform/platform.h",
    "platform/protobuf_internal.h",
    "platform/setround.h",
    "platform/snappy.h",
    "platform/tensor_coding.h",
    "platform/tracing.h",
    "util/env_var.h",
]

# Replicated for lib_internal and lib_internal_impl.
LIB_INTERNAL_DEFINES = (tf_additional_lib_defines() + [
                            "TF_USE_SNAPPY",
                        ] + tf_additional_verbs_lib_defines() +
                        tf_additional_mpi_lib_defines() +
                        tf_additional_gdr_lib_defines())

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
            "platform/*.cc",
            "platform/profile_utils/**/*.cc",
            "framework/resource_handle.cc",
            "util/env_var.cc",
        ],
        exclude = [
            "**/*test*",
            "framework/variant.cc",
            "lib/hash/crc32c_accelerate.cc",
            "lib/gif/**/*",
            "lib/jpeg/**/*",
            "lib/png/**/*",
            "platform/**/env_time.cc",
            "platform/**/cuda_libdevice_path.cc",
            "platform/**/device_tracer.cc",
            "platform/**/logger.cc",
            "platform/**/logging.cc",
            "platform/**/human_readable_json.cc",
            "platform/abi.cc",
        ],
    ) + tf_additional_lib_srcs(
        exclude = [
            "**/*test*",
            "platform/**/cuda.h",
            "platform/**/cuda_libdevice_path.cc",
            "platform/**/stream_executor.h",
            "platform/**/env_time.cc",
            "platform/**/device_tracer.cc",
            "platform/**/logger.cc",
            "platform/**/logging.cc",
            "platform/**/human_readable_json.cc",
            "platform/abi.cc",
        ] +
        # Protobuf deps already included through the ":lib_proto_parsing"
        # dependency.
        tf_additional_proto_srcs(),
    ),
    hdrs = LIB_INTERNAL_PUBLIC_HEADERS,
    copts = tf_copts(),
    defines = LIB_INTERNAL_DEFINES,
    deps = tf_additional_lib_deps() + [
        ":lib_hash_crc32c_accelerate_internal",
        ":lib_proto_parsing",
        ":abi",
        ":core_stringpiece",
        "//third_party/eigen3",
        "//tensorflow/core/platform/default/build_config:platformlib",
        "@snappy",
        "@zlib_archive//:zlib",
        "@double_conversion//:double-conversion",
        "@protobuf_archive//:protobuf",
    ] + tf_protos_all_impl() + tf_protos_grappler_impl(),
)

# File compiled with extra flags to get cpu-specific acceleration.
cc_library(
    name = "lib_hash_crc32c_accelerate_internal",
    srcs = ["lib/hash/crc32c_accelerate.cc"],
    # -msse4.2 enables the use of crc32c compiler builtins.
    copts = tf_copts() + if_linux_x86_64(["-msse4.2"]),
)

cc_library(
    name = "gif_internal",
    srcs = [
        "lib/gif/gif_io.cc",
        "platform/gif.h",
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
        "//tensorflow/core/platform/default/build_config:gif",
    ],
)

cc_library(
    name = "jpeg_internal",
    srcs = [
        "lib/jpeg/jpeg_handle.cc",
        "lib/jpeg/jpeg_mem.cc",
        "platform/jpeg.h",
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
        "//tensorflow/core/platform/default/build_config:jpeg",
    ],
)

cc_library(
    name = "png_internal",
    srcs = ["lib/png/png_io.cc"],
    hdrs = [
        "lib/bfloat16/bfloat16.h",
        "lib/core/stringpiece.h",
        "lib/png/png_io.h",
        "platform/byte_order.h",
        "platform/cpu_info.h",
        "platform/default/integral_types.h",
        "platform/default/logging.h",
        "platform/logging.h",
        "platform/macros.h",
        "platform/platform.h",
        "platform/png.h",
        "platform/types.h",
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
        "//tensorflow/core/platform/default/build_config:png",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/strings",
        "@zlib_archive//:zlib",
    ],
)

cc_library(
    name = "tflite_portable_logging",
    srcs = [
    ] + if_ios([
        "platform/default/logging.cc",
        "platform/env_time.cc",
        "platform/posix/env_time.cc",
    ]),
    hdrs = [
        "lib/bfloat16/bfloat16.h",
        "platform/default/integral_types.h",
        "platform/default/logging.h",
        "platform/logging.h",
        "platform/macros.h",
        "platform/platform.h",
        "platform/types.h",
    ] + if_windows(["platform/windows/integral_types.h"]) + if_ios(["platform/env_time.h"]),
    copts = tf_copts(),
    linkopts = ["-ldl"],
    deps = [
        "//tensorflow/core/platform/default/build_config:logging",
    ],
)

cc_library(
    name = "android_jpeg_internal",
    srcs = if_android([
        "lib/jpeg/jpeg_handle.cc",
        "lib/jpeg/jpeg_mem.cc",
        "platform/jpeg.h",
    ]),
    hdrs = [
        "lib/bfloat16/bfloat16.h",
        "lib/core/stringpiece.h",
        "lib/jpeg/jpeg_handle.h",
        "lib/jpeg/jpeg_mem.h",
        "platform/default/dynamic_annotations.h",
        "platform/default/integral_types.h",
        "platform/default/logging.h",
        "platform/dynamic_annotations.h",
        "platform/logging.h",
        "platform/macros.h",
        "platform/mem.h",
        "platform/platform.h",
        "platform/types.h",
    ],
    copts = tf_copts(),
    linkopts = ["-ldl"],
    deps = [
        "//tensorflow/core/platform/default/build_config:jpeg",
        "//tensorflow/core/platform/default/build_config:logging",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "android_gif_internal",
    srcs = if_android([
        "lib/gif/gif_io.cc",
        "platform/gif.h",
        "lib/strings/strcat.h",
        "lib/strings/numbers.h",
    ]),
    hdrs = [
        "lib/bfloat16/bfloat16.h",
        "lib/core/stringpiece.h",
        "lib/gif/gif_io.h",
        "lib/gtl/cleanup.h",
        "platform/default/dynamic_annotations.h",
        "platform/default/integral_types.h",
        "platform/default/logging.h",
        "platform/dynamic_annotations.h",
        "platform/logging.h",
        "platform/macros.h",
        "platform/mem.h",
        "platform/platform.h",
        "platform/types.h",
    ],
    copts = tf_copts(),
    linkopts = ["-ldl"],
    deps = [
        "//tensorflow/core/platform/default/build_config:gif",
        "//tensorflow/core/platform/default/build_config:logging",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "android_png_internal",
    srcs = if_android([
        "lib/png/png_io.cc",
        "platform/png.h",
    ]),
    hdrs = [
        "lib/bfloat16/bfloat16.h",
        "lib/core/stringpiece.h",
        "lib/png/png_io.h",
        "platform/byte_order.h",
        "platform/cpu_info.h",
        "platform/default/integral_types.h",
        "platform/default/logging.h",
        "platform/logging.h",
        "platform/macros.h",
        "platform/platform.h",
        "platform/types.h",
    ],
    copts = tf_copts(),
    linkopts = ["-ldl"],
    deps = [
        "//tensorflow/core/platform/default/build_config:logging",
        "@com_google_absl//absl/strings",
        "@png_archive//:png",
    ],
)

tf_proto_library(
    name = "error_codes_proto",
    srcs = ERROR_CODES_PROTO_SRCS,
    cc_api_version = 2,
    default_header = True,
    provide_cc_alias = True,
)

tf_generate_proto_text_sources(
    name = "error_codes_proto_text",
    srcs = ERROR_CODES_PROTO_SRCS,
    protodeps = [],
    srcs_relative_dir = "tensorflow/core/",
    deps = [
        ":error_codes_proto_cc",
        ":lib_internal",
    ],
)

tf_proto_library(
    name = "protos_all_proto",
    srcs = COMMON_PROTO_SRCS + ADDITIONAL_CORE_PROTO_SRCS,
    cc_api_version = 2,
    default_header = True,
    protodeps = [
        ":error_codes_proto",
    ],
)

tf_generate_proto_text_sources(
    name = "protos_all_proto_text",
    srcs = COMMON_PROTO_SRCS,
    protodeps = ERROR_CODES_PROTO_SRCS,
    srcs_relative_dir = "tensorflow/core/",
    visibility = ["//visibility:public"],
    deps = [
        ":error_codes_proto_text",
        ":lib_internal",
        ":protos_all_proto_cc",
    ],
)

cc_library(
    name = "proto_text",
    hdrs = [
        ":error_codes_proto_text_hdrs",
        ":protos_all_proto_text_hdrs",
    ],
    deps = [
        ":lib",
        ":lib_internal",
        ":protos_all_cc",
    ],
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
    "framework/tensor_reference.h",
    "framework/tracking_allocator.h",  # only needed for tests
    "framework/unique_tensor_references.h",
    "framework/variant.h",
    "util/command_line_flags.h",
    "util/equal_graph_def.h",
    "util/presized_cuckoo_map.h",
    "util/tensor_slice_set.h",
    "util/tensor_slice_util.h",
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
            "@protobuf_archive//:protobuf",
        ],
        otherwise = [
            "@protobuf_archive//:protobuf_headers",
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
        ] + [
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
            "example/example_parser_configuration.*",
            "example/feature_util.cc",
            "util/reporter.cc",
            "framework/fake_input.*",
            "framework/op_gen_lib.*",
            "framework/reader_base.*",
            "framework/resource_handle.cc",
            "util/memmapped_file_system.*",
            "util/memmapped_file_system_writer.*",
            "util/stats_calculator.*",
            "util/version_info.cc",
            "util/env_var.cc",
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
        ":feature_util",
        ":lib",
        ":lib_internal",
        ":protos_all_proto_text",
        ":error_codes_proto_text",
        ":protos_all_cc",
        ":stats_calculator_portable",
        ":version_lib",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "//tensorflow/core/platform/default/build_config:platformlib",
        "//tensorflow/core/kernels:bounds_check",
        "//third_party/eigen3",
    ] + if_static(
        extra_deps = ["@protobuf_archive//:protobuf"],
        otherwise = ["@protobuf_archive//:protobuf_headers"],
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
    srcs = ["platform/stream_executor.h"],
    hdrs = [
        "platform/cuda.h",
        "platform/stream_executor.h",
    ],
    deps = [
        "//tensorflow/core/platform/default/build_config:stream_executor",
    ],
)

# Like stream_executor library, but compiles without --config=cuda
# and does not include any cuda dependencies.
cc_library(
    name = "stream_executor_no_cuda",
    srcs = ["platform/stream_executor.h"],
    hdrs = [
        "platform/stream_executor_no_cuda.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core/platform/default/build_config:stream_executor_no_cuda",
    ],
)

tf_cuda_library(
    name = "cuda_device_functions",
    hdrs = ["util/cuda_device_functions.h"],
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
        ":proto_text",
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
    srcs = [
        "common_runtime/eval_const_tensor.cc",
        "common_runtime/scoped_allocator.cc",
        "common_runtime/scoped_allocator_mgr.cc",
        "common_runtime/shape_refiner.cc",
        "common_runtime/shape_refiner.h",
        "framework/versions.h",
        "graph/graph_constructor.cc",  # Depends on common_runtime.
        "graph/graph_def_builder_util.cc",  # Depends on common_runtime.
        "public/session.h",
        "public/session_options.h",
        "public/version.h",
    ],
    hdrs = CORE_CPU_BASE_HDRS,
    copts = tf_copts(),
    deps = [
        ":graph",
        ":framework",
        ":framework_internal",
        ":lib",
        ":lib_internal",
        ":proto_text",
        ":protos_all_cc",
        "//third_party/eigen3",
    ] + if_static([
        ":function_ops_op_lib",
        ":functional_grad",
        ":functional_ops_op_lib",
        "//tensorflow/core/kernels:bounds_check",
        "//tensorflow/core/kernels:required",
    ]),
    alwayslink = 1,
)

CORE_CPU_LIB_HEADERS = CORE_CPU_BASE_HDRS + [
    "common_runtime/allocator_retry.h",
    "common_runtime/base_collective_executor.h",
    "common_runtime/bfc_allocator.h",
    "common_runtime/hierarchical_tree_broadcaster.h",
    "common_runtime/buf_rendezvous.h",
    "common_runtime/build_graph_options.h",
    "common_runtime/collective_executor_mgr.h",
    "common_runtime/collective_param_resolver_local.h",
    "common_runtime/collective_rma_local.h",
    "common_runtime/collective_util.h",
    "common_runtime/constant_folding.h",
    "common_runtime/copy_tensor.h",
    "common_runtime/costmodel_manager.h",
    "common_runtime/debugger_state_interface.h",
    "common_runtime/device_resolver_local.h",
    "common_runtime/dma_helper.h",
    "common_runtime/eigen_thread_pool.h",
    "common_runtime/executor.h",
    "common_runtime/executor_factory.h",
    "common_runtime/graph_optimizer.h",
    "common_runtime/local_device.h",
    "common_runtime/lower_if_op.h",
    "common_runtime/lower_if_while.h",
    "common_runtime/lower_while_op.h",
    "common_runtime/memory_types.h",
    "common_runtime/mkl_cpu_allocator.h",
    "common_runtime/optimization_registry.h",
    "common_runtime/pending_counts.h",
    "common_runtime/partitioning_utils.h",
    "common_runtime/placer.h",
    "common_runtime/process_util.h",
    "common_runtime/profile_handler.h",
    "common_runtime/renamed_device.h",
    "common_runtime/rendezvous_mgr.h",
    "common_runtime/rendezvous_util.h",
    "common_runtime/ring_reducer.h",
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
        "common_runtime/allocator_retry.cc",
        "common_runtime/base_collective_executor.cc",
        "common_runtime/bfc_allocator.cc",
        "common_runtime/buf_rendezvous.cc",
        "common_runtime/build_graph_options.cc",
        "common_runtime/collective_executor_mgr.cc",
        "common_runtime/collective_param_resolver_local.cc",
        "common_runtime/collective_rma_local.cc",
        "common_runtime/collective_util.cc",
        "common_runtime/constant_folding.cc",
        "common_runtime/copy_tensor.cc",
        "common_runtime/costmodel_manager.cc",
        "common_runtime/debugger_state_interface.cc",
        "common_runtime/device.cc",
        "common_runtime/device_factory.cc",
        "common_runtime/device_mgr.cc",
        "common_runtime/device_resolver_local.cc",
        "common_runtime/device_set.cc",
        "common_runtime/executor.cc",
        "common_runtime/executor_factory.cc",
        "common_runtime/function.cc",
        "common_runtime/graph_optimizer.cc",
        "common_runtime/graph_runner.cc",
        "common_runtime/hierarchical_tree_broadcaster.cc",
        "common_runtime/local_device.cc",
        "common_runtime/lower_if_op.cc",
        "common_runtime/lower_if_while.cc",
        "common_runtime/lower_while_op.cc",
        "common_runtime/memory_types.cc",
        "common_runtime/mkl_cpu_allocator.cc",
        "common_runtime/optimization_registry.cc",
        "common_runtime/parallel_concat_optimizer.cc",
        "common_runtime/partitioning_utils.cc",
        "common_runtime/placer.cc",
        "common_runtime/pool_allocator.cc",
        "common_runtime/process_function_library_runtime.cc",
        "common_runtime/process_state.cc",
        "common_runtime/process_util.cc",
        "common_runtime/renamed_device.cc",
        "common_runtime/rendezvous_mgr.cc",
        "common_runtime/rendezvous_util.cc",
        "common_runtime/ring_reducer.cc",
        "common_runtime/session.cc",
        "common_runtime/session_factory.cc",
        "common_runtime/session_options.cc",
        "common_runtime/session_state.cc",
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
    copts = tf_copts(),
    deps = [
        ":graph",
        ":framework",
        ":framework_internal",
        ":lib",
        ":lib_internal",
        ":proto_text",
        ":protos_all_cc",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "//third_party/eigen3",
        "//tensorflow/core/grappler/utils:functions",
    ] + mkl_deps(),
    alwayslink = 1,
)

tf_cuda_library(
    name = "core_cpu_lib",
    hdrs = CORE_CPU_LIB_HEADERS,
    deps = [
        ":core_cpu_base",
        ":proto_text",
        "//tensorflow/core/grappler:grappler_item",
    ] + if_static([":core_cpu_impl"]) + tf_protos_all() + tf_protos_grappler(),
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
        ":metrics",
        ":proto_text",
        ":protos_all_cc",
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

cc_library(
    name = "regexp_internal",
    hdrs = [
        "platform/regexp.h",
    ],
    visibility = [
        "//tensorflow/compiler:__subpackages__",
        "//tensorflow/core/kernels:__subpackages__",
        "//tensorflow/core/profiler:__subpackages__",
    ],
    deps = [":lib_internal"],
)

tf_cuda_library(
    name = "metrics",
    srcs = ["common_runtime/metrics.cc"],
    hdrs = ["common_runtime/metrics.h"],
    deps = [
        ":lib",
    ],
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
        ":device_tracer",
        ":framework",
        ":framework_internal",
        ":graph",
        ":lib",
        ":lib_internal",
        ":metrics",
        ":proto_text",
        ":protos_all_cc",
        "//tensorflow/core/debug:debug_graph_utils",
        "//tensorflow/core/kernels:function_ops",
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
        ":proto_text",
        ":protos_all_cc",
    ],
    alwayslink = 1,
)

tf_cuda_library(
    name = "device_tracer",
    srcs = tf_additional_device_tracer_srcs(),
    hdrs = [
        "platform/device_tracer.h",
    ],
    copts = tf_copts(),
    cuda_deps = if_cuda_is_configured(tf_additional_cupti_wrapper_deps() + tf_additional_device_tracer_cuda_deps()),
    visibility = [
        "//tensorflow:internal",
    ],
    deps = [
        ":core_cpu_internal",
        ":lib",
        ":protos_all_cc",
    ] + tf_additional_device_tracer_deps(),
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
    "common_runtime/gpu/cuda_host_allocator.h",
    "common_runtime/gpu/gpu_bfc_allocator.h",
    "common_runtime/gpu/gpu_cudamalloc_allocator.h",
    "common_runtime/gpu/gpu_debug_allocator.h",
    "common_runtime/gpu/gpu_device.h",
    "common_runtime/gpu/gpu_id.h",
    "common_runtime/gpu/gpu_id_manager.h",
    "common_runtime/gpu/gpu_id_utils.h",
    "common_runtime/gpu/gpu_init.h",
    "common_runtime/gpu/gpu_managed_allocator.h",
    "common_runtime/gpu/gpu_process_state.h",
    "common_runtime/gpu/gpu_stream_util.h",
    "common_runtime/gpu/gpu_util.h",
    "common_runtime/gpu_device_context.h",
]

tf_cuda_library(
    name = "gpu_runtime_impl",
    srcs = [
        "common_runtime/gpu/gpu_bfc_allocator.cc",
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
        ":proto_text",
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
        "lib/gtl/manual_constructor.h",
        "lib/io/block.h",
        "lib/io/block_builder.h",
        "lib/io/format.h",
        "lib/random/philox_random_test_utils.h",
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
    srcs = ["platform/test_main.cc"],
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

# This is the lite version of a main() for tests. It does not include any
# support for reporting benchmark results when running on TPUs.
cc_library(
    name = "test_lite_main",
    testonly = 1,
    srcs = ["platform/test_main.cc"],
    copts = tf_copts(),
    deps = [
        # TODO(ahentz): we don't want to depend on "lib" here. It used to be
        # that "core_stringpiece" was enough but that recently changed and
        # we now need at least "str_util".
        ":lib",
        ":lib_platform",
        ":stacktrace_handler",
        ":test_lite",
        "//tensorflow/core/platform/default/build_config:test_lite_main",
    ],
    alwayslink = 1,
)

tf_cc_tests(
    name = "low_level_library_tests",
    size = "small",
    srcs = [
        "lib/core/arena_test.cc",
        "lib/core/bitmap_test.cc",
        "lib/core/blocking_counter_test.cc",
        "lib/core/coding_test.cc",
        "lib/core/notification_test.cc",
        "lib/core/refcount_test.cc",
        "lib/core/status_test.cc",
        "lib/core/stringpiece_test.cc",
        "lib/core/threadpool_test.cc",
        "lib/gtl/cleanup_test.cc",
        "lib/gtl/compactptrset_test.cc",
        "lib/gtl/edit_distance_test.cc",
        "lib/gtl/flatmap_test.cc",
        "lib/gtl/flatset_test.cc",
        "lib/gtl/int_type_test.cc",
        "lib/gtl/iterator_range_test.cc",
        "lib/gtl/manual_constructor_test.cc",
        "lib/gtl/map_util_test.cc",
        "lib/gtl/top_n_test.cc",
        "lib/hash/crc32c_test.cc",
        "lib/hash/hash_test.cc",
        "lib/histogram/histogram_test.cc",
        "lib/io/buffered_inputstream_test.cc",
        "lib/io/inputbuffer_test.cc",
        "lib/io/inputstream_interface_test.cc",
        "lib/io/path_test.cc",
        "lib/io/random_inputstream_test.cc",
        "lib/io/record_reader_writer_test.cc",
        "lib/io/recordio_test.cc",
        "lib/io/snappy/snappy_buffers_test.cc",
        "lib/io/table_test.cc",
        "lib/io/zlib_buffers_test.cc",
        "lib/math/math_util_test.cc",
        "lib/monitoring/collection_registry_test.cc",
        "lib/monitoring/counter_test.cc",
        "lib/monitoring/gauge_test.cc",
        "lib/monitoring/metric_def_test.cc",
        "lib/monitoring/sampler_test.cc",
        "lib/random/distribution_sampler_test.cc",
        "lib/random/philox_random_test.cc",
        "lib/random/random_test.cc",
        "lib/random/simple_philox_test.cc",
        "lib/strings/base64_test.cc",
        "lib/strings/numbers_test.cc",
        "lib/strings/scanner_test.cc",
        "lib/strings/str_util_test.cc",
        "lib/strings/strcat_test.cc",
        "lib/strings/stringprintf_test.cc",
        "lib/wav/wav_io_test.cc",
        "platform/fingerprint_test.cc",
        "platform/integral_types_test.cc",
        "platform/logging_test.cc",
        "platform/mutex_test.cc",
        "platform/net_test.cc",
        "platform/port_test.cc",
        "platform/profile_utils/cpu_utils_test.cc",
        "platform/stacktrace_handler_test.cc",
        "platform/subprocess_test.cc",
        "platform/vmodule_benchmark_test.cc",
    ],
    deps = [
        ":lib",
        ":lib_internal",
        ":lib_test_internal",
        ":protos_all_cc",
        ":test",
        ":test_main",
        "//third_party/eigen3",
        "@zlib_archive//:zlib",
    ],
)

tf_cc_test(
    name = "vmodule_test",
    srcs = ["platform/vmodule_test.cc"],
    tags = ["optonly"],
    deps = [
        ":lib",
        ":lib_internal",
        ":lib_test_internal",
        ":protos_all_cc",
        ":test",
        "//third_party/eigen3",
    ],
)

tf_cc_test(
    name = "lib_random_random_distributions_test",
    srcs = ["lib/random/random_distributions_test.cc"],
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
    srcs = ["platform/platform_strings_test.cc"],
    deps = [
        ":lib",
        ":platform_strings",
    ],
)

tf_cc_test(
    name = "platform_env_test",
    size = "small",
    srcs = ["platform/env_test.cc"],
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
    srcs = ["platform/fake_python_env_test.cc"],
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
    srcs = ["platform/abi_test.cc"],
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
    srcs = ["platform/numa_test.cc"],
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
    srcs = ["platform/setround_test.cc"],
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
    srcs = ["platform/file_system_test.cc"],
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
    srcs = ["lib/strings/ordered_code_test.cc"],
    extra_copts = ["$(STACK_FRAME_UNLIMITED)"],  # Tests initialize large vectors
    deps = [
        ":lib",
        ":lib_internal",
        ":test",
        ":test_main",
    ],
)

tf_cc_test(
    name = "lib_random_weighted_picker_test",
    size = "medium",
    srcs = ["lib/random/weighted_picker_test.cc"],
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
        "common_runtime/optimization_registry_test.cc",
        "common_runtime/pending_counts_test.cc",
        "common_runtime/placer_test.cc",
        "common_runtime/session_test.cc",
        "example/feature_util_test.cc",
        "framework/allocator_test.cc",
        "framework/attr_value_util_test.cc",
        "framework/bfloat16_test.cc",
        "framework/cancellation_test.cc",
        "framework/common_shape_fns_test.cc",
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
        "util/device_name_utils_test.cc",
        "util/dump_graph_test.cc",
        "util/equal_graph_def_test.cc",
        "util/events_writer_test.cc",
        "util/example_proto_fast_parsing_test.cc",
        "util/example_proto_helper_test.cc",
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
        "//tensorflow:darwin": ["-headerpad_max_install_names"],
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
        "//tensorflow:darwin": ["-headerpad_max_install_names"],
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
        ":cudnn_rnn_ops",
        "//tensorflow/core",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
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
    tags = tf_cuda_tests_tags(),
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
    tags = tf_cuda_tests_tags(),
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
        "//tensorflow/core/kernels:mkl_concat_op",
        "//tensorflow/core/kernels:mkl_conv_op",
        "//tensorflow/core/kernels:mkl_cwise_ops_common",
        "//tensorflow/core/kernels:mkl_fused_batch_norm_op",
        "//tensorflow/core/kernels:mkl_identity_op",
        "//tensorflow/core/kernels:mkl_input_conversion_op",
        "//tensorflow/core/kernels:mkl_lrn_op",
        "//tensorflow/core/kernels:mkl_pooling_ops",
        "//tensorflow/core/kernels:mkl_relu_op",
        "//tensorflow/core/kernels:mkl_reshape_op",
        "//tensorflow/core/kernels:mkl_slice_op",
        "//tensorflow/core/kernels:mkl_softmax_op",
        "//tensorflow/core/kernels:mkl_tfconv_op",
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

tf_cuda_only_cc_test(
    name = "util_cuda_kernel_helper_test",
    srcs = [
        "util/cuda_kernel_helper_test.cu.cc",
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
    tags = tf_cuda_tests_tags(),
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
    deps = [
        ":core_cpu",
        ":core_cpu_internal",
        ":framework",
        ":framework_internal",
        ":lib",
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
        "//third_party/eigen3",
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
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:scope",
        "//tensorflow/core/kernels:cwise_op",
        "//third_party/eigen3",
    ],
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
    args = ["\"--gtest_death_test_style=threadsafe\""],
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
        "ops/set_ops_test.cc",
        "ops/shape_function_test.cc",
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

tf_cc_test_gpu(
    name = "device_tracer_test",
    size = "small",
    srcs = ["platform/device_tracer_test.cc"],
    args =
        ["--heap_check=local"] + tf_additional_device_tracer_test_flags(),
    linkstatic = tf_kernel_tests_linkstatic(),
    tags = tf_cuda_tests_tags() + ["nomac"],
    deps = [
        ":all_kernels",
        ":core_cpu",
        ":core_cpu_internal",
        ":device_tracer",
        ":direct_session",
        ":direct_session_internal",
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
        "//tensorflow/core/kernels:ops_util",
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
    ],
)

tf_cc_tests(
    name = "common_runtime_lower_if_while_test",
    size = "small",
    srcs = ["common_runtime/lower_if_while_test.cc"],
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
        #   0 : 'a'
        #   1 : 'b'
        #    ...
        #   9 : 'j'
        "lib/lmdb/testdata/data.mdb",
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
    srcs = tf_additional_libdevice_srcs(),
    hdrs = ["platform/cuda_libdevice_path.h"],
    copts = tf_copts(),
    data = tf_additional_libdevice_data(),
    visibility = ["//visibility:public"],
    deps = [
        ":lib",
    ] + tf_additional_libdevice_deps(),
)

transitive_hdrs(
    name = "headers",
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:platform_strings",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:stream_executor",
    ],
)

# -----------------------------------------------------------------------------
# Google-internal targets go here (must be at the end).

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

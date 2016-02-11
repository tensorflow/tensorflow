# Description:
# TensorFlow is a computational framework, primarily for use in machine
# learning applications.
#
# Public targets:
#
# ":protos_all" - exports all core TensorFlow protos
#     ":protos_all_py_pb2" - py_proto_library version (Google-internal)
# ":lib" - exports the public non-test headers for:
#     platform/: Platform-specific code and external dependencies
#     lib/: Low-level libraries that are not TensorFlow-specific
# ":test" - test equivalent of ":lib".
#     This is currently public, but may be made internal in the
#     future.  Try to avoid depending on it.
# ":framework" - exports the public non-test headers for:
#     util/: General low-level TensorFlow-specific libraries
#     framework/: Support for adding new ops & kernels
# ":ops" - defines TensorFlow ops, but no implementations / kernels
#     ops/: Standard ops
#     user_ops/: User-supplied ops
#     This aggregates a number of smaller op libraries (":*_op_lib")
# ":core_cpu" - exports the public non-test headers for:
#     graph/: Support for graphs made up of ops
#     common_runtime/: Common code for execution of graphs
#     public/: Public APIs for running graphs
# ":core" - The code for ":core_cpu" plus a GPU runtime
# ":kernel_lib" - Library with code common to multiple kernel implementations.
# ":gpu_kernels" - The gpu-specific kernels
# ":all_kernels" - The cpu-specific kernels, plus ":gpu_kernels" if
#     built with Cuda
# ":tensorflow_opensource" - The complete open-source package, including
#      ":all_kernels", ":core", and a Session implementation.
# ":tensorflow" - "tensorflow_opensource" plus some Google-internal libraries.
# ":testlib" - TensorFlow-specific test support, e.g. utilities for testing
#      kernels.
#
# Public Android targets:
#
# filegroup ":android_proto_srcs" - Protos
# filegroup ":android_srcs" - Core sources
# filegroup ":android_core_ops" - Essential kernels
# filegroup ":android_extended_ops" - Optional kernels
# filegroup ":android_extended_ops_group1" - Optional kernels, first batch
# filegroup ":android_extended_ops_group2" - Optional kernels, second batch
# cc_library ":android_tensorflow_lib" - Native library
# portable_proto_library ":android_proto_lib" (Google-internal)

package(default_visibility = ["//tensorflow:internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "tf_copts")
load("//tensorflow:tensorflow.bzl", "tf_cc_tests")
load("//tensorflow:tensorflow.bzl", "tf_cuda_library")
load("//tensorflow:tensorflow.bzl", "tf_gen_op_libs")
load("//tensorflow:tensorflow.bzl", "tf_gpu_kernel_library")

# For platform specific build config
load(
    "//tensorflow/core:platform/default/build_config.bzl",
    "tf_proto_library",
    "tf_additional_lib_srcs",
    "tf_additional_stream_executor_srcs",
    "tf_additional_test_srcs",
    "tf_kernel_tests_linkstatic",
)
load(
    "//tensorflow/core:platform/default/build_config_root.bzl",
    "tf_cuda_tests_tags",
)

# -----------------------------------------------------------------------------
# Public targets

tf_proto_library(
    name = "protos_all",
    srcs = glob(["**/*.proto"]),
    cc_api_version = 2,
    go_api_version = 2,
    java_api_version = 2,
    py_api_version = 2,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "lib",
    hdrs = [
        # TODO(josh11b): Make many of these internal.
        "lib/core/arena.h",
        "lib/core/bits.h",
        "lib/core/casts.h",
        "lib/core/coding.h",
        "lib/core/command_line_flags.h",  # TODO(vrv): Delete.
        "lib/core/errors.h",
        "lib/core/notification.h",
        "lib/core/status.h",
        "lib/core/stringpiece.h",
        "lib/core/threadpool.h",
        "lib/gtl/array_slice.h",
        "lib/gtl/inlined_vector.h",
        "lib/gtl/map_util.h",  # TODO(josh11b): make internal
        "lib/gtl/stl_util.h",  # TODO(josh11b): make internal
        "lib/hash/crc32c.h",  # TODO(josh11b): make internal
        "lib/hash/hash.h",  # TODO(josh11b): make internal
        "lib/histogram/histogram.h",
        "lib/io/inputbuffer.h",  # TODO(josh11b): make internal
        "lib/io/path.h",
        "lib/io/record_reader.h",
        "lib/io/record_writer.h",
        "lib/io/table.h",
        "lib/io/table_builder.h",
        "lib/io/table_options.h",
        "lib/jpeg/jpeg_mem.h",
        "lib/random/distribution_sampler.h",
        "lib/random/philox_random.h",
        "lib/random/simple_philox.h",  # TODO(josh11b): make internal
        "lib/strings/numbers.h",  # TODO(josh11b): make internal
        "lib/strings/str_util.h",  # TODO(josh11b): make internal
        "lib/strings/strcat.h",
        "lib/strings/stringprintf.h",
        "platform/env.h",
        "platform/host_info.h",  # TODO(josh11b): make internal
        "platform/init_main.h",
        "platform/logging.h",
        "platform/macros.h",
        "platform/mem.h",
        "platform/mutex.h",
        "platform/protobuf.h",  # TODO(josh11b): make internal
        "platform/regexp.h",
        "platform/thread_annotations.h",
        "platform/types.h",
    ],
    visibility = ["//visibility:public"],
    deps = [":lib_internal"],
)

# Test support library needed for all tests
# This is currently public, but may be made internal in the
# future.  Try to avoid depending on it.
cc_library(
    name = "test",
    testonly = 1,
    srcs = [
        "platform/test.cc",
    ] + tf_additional_test_srcs(),
    hdrs = [
        "lib/core/status_test_util.h",
        "platform/test.h",
        "platform/test_benchmark.h",
    ],
    copts = tf_copts(),
    linkopts = ["-lm"],
    visibility = ["//visibility:public"],
    deps = [
        ":lib",
        ":lib_internal",
        "//tensorflow/core/platform/default/build_config:gtest",
    ],
)

tf_cuda_library(
    name = "framework",
    hdrs = [
        "framework/allocator.h",
        "framework/attr_value_util.h",
        "framework/bfloat16.h",
        "framework/cancellation.h",
        "framework/control_flow.h",  # TODO(josh11b): Make internal?
        "framework/device_base.h",
        "framework/function.h",
        "framework/kernel_def_builder.h",
        "framework/lookup_interface.h",
        "framework/memory_types.h",
        "framework/node_def_builder.h",
        "framework/node_def_util.h",
        "framework/numeric_op.h",
        "framework/numeric_types.h",
        "framework/op.h",
        "framework/op_def_builder.h",
        "framework/op_def_util.h",
        "framework/op_gen_lib.h",
        "framework/op_kernel.h",
        "framework/partial_tensor_shape.h",
        "framework/queue_interface.h",
        "framework/reader_interface.h",
        "framework/reader_op_kernel.h",
        "framework/register_types.h",
        "framework/resource_mgr.h",
        "framework/tensor.h",
        "framework/tensor_shape.h",
        "framework/tensor_slice.h",
        "framework/tensor_types.h",
        "framework/tensor_util.h",
        "framework/type_index.h",
        "framework/type_traits.h",
        "framework/types.h",
        "public/version.h",
        "util/bcast.h",
        "util/cuda_kernel_helper.h",
        "util/device_name_utils.h",
        "util/events_writer.h",
        "util/guarded_philox_random.h",
        "util/padding.h",
        "util/port.h",
        "util/saved_tensor_slice_util.h",
        "util/sparse/group_iterator.h",
        "util/sparse/sparse_tensor.h",
        "util/tensor_slice_reader.h",
        "util/tensor_slice_reader_cache.h",
        "util/tensor_slice_writer.h",
        "util/use_cudnn.h",
        "util/util.h",
        "util/work_sharder.h",
    ],
    visibility = ["//visibility:public"],
    deps = [":framework_internal"],
)

cc_binary(
    name = "libtensorflow_framework.so",
    srcs = [],
    linkopts = select({
        "//conditions:default": [
            "-Wl,-Bsymbolic",
            "-lm",
        ],
        "//tensorflow:darwin": [],
    }),
    linkshared = 1,
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [":framework"],
)

tf_cuda_library(
    name = "ops",
    srcs = glob(
        [
            "ops/**/*.h",
            "ops/**/*.cc",
            "user_ops/**/*.h",
            "user_ops/**/*.cc",
        ],
        exclude = [
            "**/*test*",
            "**/*main.cc",
            "user_ops/**/*.cu.cc",
        ],
    ),
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":framework",
        ":lib",
        ":protos_cc",
        "//tensorflow/models/embedding:word2vec_ops",
        "//third_party/eigen3",
    ],
    alwayslink = 1,
)

# Generates library per group of ops.
tf_gen_op_libs(
    op_lib_names = [
        "array_ops",
        "attention_ops",
        "candidate_sampling_ops",
        "control_flow_ops",
        "data_flow_ops",
        "image_ops",
        "io_ops",
        "linalg_ops",
        "logging_ops",
        "functional_ops",
        "math_ops",
        "nn_ops",
        "no_op",
        "parsing_ops",
        "random_ops",
        "script_ops",
        "sendrecv_ops",
        "sparse_ops",
        "state_ops",
        "string_ops",
        "summary_ops",
        "training_ops",
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

tf_cuda_library(
    name = "core_cpu",
    hdrs = [
        "common_runtime/device.h",
        "graph/algorithm.h",
        "graph/default_device.h",
        "graph/graph.h",
        "graph/graph_constructor.h",
        "graph/graph_def_builder.h",
        "graph/node_builder.h",
        "graph/validate.h",
        "public/session.h",
        "public/session_options.h",
        "public/tensor_c_api.h",
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
    ],
)

tf_gpu_kernel_library(
    name = "gpu_kernels",
    srcs = glob(
        [
            "kernels/**/*.h",
            "kernels/*.cu.cc",
            "user_ops/**/*.h",
            "user_ops/*.cu.cc",
        ],
    ),
    visibility = ["//visibility:public"],
    deps = [
        ":cuda",
        ":framework",
        ":framework_internal",
        ":lib",
        ":lib_internal",
        "//third_party/eigen3",
    ],
)

# Common libraries used for implementing kernels.
# TODO(josh11b): This also currently includes all the kernel implementations,
# should split that out into the ":all_kernels" target.
tf_cuda_library(
    name = "kernel_lib",
    # TODO(josh11b): Pull common dependencies out into a separate library.
    srcs = glob(
        [
            "kernels/**/*.cc",
        ],
        exclude = [
            "kernels/**/*test*",
            "kernels/**/*main.cc",
            "kernels/**/*.cu.cc",
        ],
    ),
    hdrs = glob(["kernels/**/*.h"]),
    copts = tf_copts(),
    cuda_deps = [
        ":gpu_kernels",
    ],
    linkstatic = 0,
    visibility = ["//visibility:public"],
    deps = [
        ":core_cpu",
        ":framework",
        ":framework_internal",
        ":lib",
        ":lib_internal",
        ":ops",
        ":protos_all_cc",
        ":stream_executor",
        "//tensorflow/models/embedding:word2vec_kernels",
        "//third_party/eigen3",
    ],
    alwayslink = 1,
)

# This includes implementations of all kernels built into TensorFlow.
tf_cuda_library(
    name = "all_kernels",
    visibility = ["//visibility:public"],
    deps = [
        ":kernel_lib",
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
        ":gpu_runtime",
        ":lib",
    ],
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
        "common_runtime/kernel_benchmark_testlib.cc",
        "framework/fake_input.cc",
        "framework/function_testlib.cc",
        "framework/tensor_testutil.cc",
        "graph/testlib.cc",
        "kernels/ops_testutil.cc",
    ],
    hdrs = [
        "common_runtime/kernel_benchmark_testlib.h",
        "framework/fake_input.h",
        "framework/function_testlib.h",
        "framework/tensor_testutil.h",
        "graph/testlib.h",
        "kernels/ops_testutil.h",
    ],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":all_kernels",
        ":core_cpu",
        ":core_cpu_internal",
        ":framework",
        ":framework_internal",
        ":kernel_lib",
        ":lib",
        ":protos_all_cc",
        ":tensorflow",  # TODO(josh11b): Drop this dep?
        ":test",
        "//tensorflow/core/platform/default/build_config:gtest",
    ],
)

# -----------------------------------------------------------------------------
# Public Android targets

# Android-specific BUILD targets
load("//tensorflow:tensorflow.bzl", "tf_android_core_proto_sources")

# List of protos we want on android
filegroup(
    name = "android_proto_srcs",
    srcs = tf_android_core_proto_sources(),
    visibility = ["//visibility:public"],
)

# Core sources. Should eventually become identical to open source
# sources.
filegroup(
    name = "android_srcs",
    srcs = glob(
        [
            "client/**/*.cc",
            "common_runtime/**/*.h",
            "common_runtime/**/*.cc",
            "framework/**/*.h",
            "framework/**/*.cc",
            "graph/**/*.h",
            "graph/**/*.cc",
            "lib/**/*.h",
            "lib/**/*.cc",
            "ops/**/*.cc",
            "ops/**/*.h",
            "platform/*.h",
            "platform/*.cc",
            "platform/**/*.h",
            "platform/**/*.cc",
            "public/**/*.h",
            "util/**/*.h",
            "util/**/*.cc",
            "kernels/ops_util.cc",
            "kernels/ops_util.h",
            "kernels/avgpooling_op.h",
            "kernels/maxpooling_op.h",
            "kernels/pooling_ops_common.h",
            "kernels/pooling_ops_common.cc",
        ],
        exclude = [
            "**/*test.cc",
            "**/*testutil*",
            "**/*testlib*",
            "**/*main.cc",
            "lib/jpeg/*.h",
            "lib/jpeg/*.cc",
            "lib/png/*.h",
            "lib/png/*.cc",
            "util/events_writer.cc",
            "util/events_writer.h",
            # Exclude all protobuf/google headers except protobuf_android.h
            "platform/google/cord_coding.h",
            "platform/google/dynamic_annotations.h",
            "platform/google/integral_types.h",
            "platform/google/mutex.h",
            "platform/google/protobuf.h",
            "platform/google/stream_executor.h",
            "platform/google/tracing_impl.h",
            "platform/google/*.cc",
            "platform/google/test_benchmark.cc",
            "platform/google/test_benchmark.h",
            "kernels/**/*.cu.cc",
            "user_ops/**/*.cu.cc",
            "common_runtime/gpu/*.cc",
            "common_runtime/gpu_device_factory.cc",
        ],
    ),
    visibility = ["//visibility:public"],
)

# Core kernels we want on Android. Only a subset of kernels to keep
# base library small.
filegroup(
    name = "android_core_ops",
    srcs = [
        "//tensorflow/core:kernels/aggregate_ops.cc",
        "//tensorflow/core:kernels/aggregate_ops.h",
        "//tensorflow/core:kernels/aggregate_ops_cpu.h",
        "//tensorflow/core:kernels/assign_op.h",
        "//tensorflow/core:kernels/bias_op.cc",
        "//tensorflow/core:kernels/bias_op.h",
        "//tensorflow/core:kernels/cast_op.cc",
        "//tensorflow/core:kernels/cast_op.h",
        "//tensorflow/core:kernels/concat_op.cc",
        "//tensorflow/core:kernels/concat_op.h",
        "//tensorflow/core:kernels/concat_op_cpu.cc",
        "//tensorflow/core:kernels/constant_op.cc",
        "//tensorflow/core:kernels/constant_op.h",
        "//tensorflow/core:kernels/cwise_ops.h",
        "//tensorflow/core:kernels/cwise_ops_common.cc",
        "//tensorflow/core:kernels/cwise_ops_common.h",
        "//tensorflow/core:kernels/dense_update_ops.cc",
        "//tensorflow/core:kernels/dense_update_ops.h",
        "//tensorflow/core:kernels/example_parsing_ops.cc",
        "//tensorflow/core:kernels/fill_functor.h",
        "//tensorflow/core:kernels/gather_op.cc",
        "//tensorflow/core:kernels/identity_op.cc",
        "//tensorflow/core:kernels/identity_op.h",
        "//tensorflow/core:kernels/matmul_op.cc",
        "//tensorflow/core:kernels/matmul_op.h",
        "//tensorflow/core:kernels/no_op.cc",
        "//tensorflow/core:kernels/no_op.h",
        "//tensorflow/core:kernels/pack_op.cc",
        "//tensorflow/core:kernels/reshape_op.cc",
        "//tensorflow/core:kernels/reshape_op.h",
        "//tensorflow/core:kernels/reverse_sequence_op.cc",
        "//tensorflow/core:kernels/reverse_sequence_op.h",
        "//tensorflow/core:kernels/sendrecv_ops.cc",
        "//tensorflow/core:kernels/sendrecv_ops.h",
        "//tensorflow/core:kernels/sequence_ops.cc",
        "//tensorflow/core:kernels/shape_ops.cc",
        "//tensorflow/core:kernels/slice_op.cc",
        "//tensorflow/core:kernels/slice_op.h",
        "//tensorflow/core:kernels/softmax_op.cc",
        "//tensorflow/core:kernels/softmax_op.h",
        "//tensorflow/core:kernels/split_op.cc",
        "//tensorflow/core:kernels/split_op.h",
        "//tensorflow/core:kernels/split_op_cpu.cc",
        "//tensorflow/core:kernels/unpack_op.cc",
        "//tensorflow/core:kernels/variable_ops.cc",
        "//tensorflow/core:kernels/variable_ops.h",
    ],
    visibility = ["//visibility:public"],
)

# Other kernels we may want on Android.
#
# The kernels can be consumed as a whole or in two groups for
# supporting separate compilation. Note that the split into groups
# is entirely for improving compilation time, and not for
# organizational reasons; you should not depend on any
# of those groups independently.
filegroup(
    name = "android_extended_ops",
    srcs = [
        ":android_extended_ops_group1",
        ":android_extended_ops_group2",
    ],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "android_extended_ops_group1",
    srcs = [
        "//tensorflow/core:kernels/argmax_op.cc",
        "//tensorflow/core:kernels/argmax_op.h",
        "//tensorflow/core:kernels/avgpooling_op.cc",
        "//tensorflow/core:kernels/avgpooling_op.h",
        "//tensorflow/core:kernels/bcast_ops.cc",
        "//tensorflow/core:kernels/control_flow_ops.cc",
        "//tensorflow/core:kernels/control_flow_ops.h",
        "//tensorflow/core:kernels/conv_2d.h",
        "//tensorflow/core:kernels/conv_ops.cc",
        "//tensorflow/core:kernels/cwise_op_add.cc",
        "//tensorflow/core:kernels/cwise_op_div.cc",
        "//tensorflow/core:kernels/cwise_op_equal_to.cc",
        "//tensorflow/core:kernels/cwise_op_exp.cc",
        "//tensorflow/core:kernels/cwise_op_greater.cc",
        "//tensorflow/core:kernels/cwise_op_isfinite.cc",
        "//tensorflow/core:kernels/cwise_op_less.cc",
        "//tensorflow/core:kernels/cwise_op_log.cc",
        "//tensorflow/core:kernels/cwise_op_maximum.cc",
        "//tensorflow/core:kernels/cwise_op_minimum.cc",
        "//tensorflow/core:kernels/cwise_op_mul.cc",
        "//tensorflow/core:kernels/cwise_op_neg.cc",
        "//tensorflow/core:kernels/cwise_op_select.cc",
        "//tensorflow/core:kernels/cwise_op_sigmoid.cc",
        "//tensorflow/core:kernels/cwise_op_sqrt.cc",
        "//tensorflow/core:kernels/cwise_op_square.cc",
        "//tensorflow/core:kernels/cwise_op_sub.cc",
        "//tensorflow/core:kernels/cwise_op_tanh.cc",
        "//tensorflow/core:kernels/dynamic_partition_op.cc",
        "//tensorflow/core:kernels/dynamic_stitch_op.cc",
        "//tensorflow/core:kernels/in_topk_op.cc",
        "//tensorflow/core:kernels/io.cc",
        "//tensorflow/core:kernels/io.h",
        "//tensorflow/core:kernels/lrn_op.cc",
    ],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "android_extended_ops_group2",
    srcs = [
        "//tensorflow/core:kernels/io.h",
        "//tensorflow/core:kernels/maxpooling_op.cc",
        "//tensorflow/core:kernels/maxpooling_op.h",
        "//tensorflow/core:kernels/reduction_ops.h",
        "//tensorflow/core:kernels/reduction_ops_common.h",
        "//tensorflow/core:kernels/reduction_ops_max.cc",
        "//tensorflow/core:kernels/reduction_ops_mean.cc",
        "//tensorflow/core:kernels/reduction_ops_min.cc",
        "//tensorflow/core:kernels/reduction_ops_prod.cc",
        "//tensorflow/core:kernels/reduction_ops_sum.cc",
        "//tensorflow/core:kernels/relu_op.cc",
        "//tensorflow/core:kernels/relu_op.h",
        "//tensorflow/core:kernels/restore_op.cc",
        "//tensorflow/core:kernels/save_op.cc",
        "//tensorflow/core:kernels/softplus_op.cc",
        "//tensorflow/core:kernels/softplus_op.h",
        "//tensorflow/core:kernels/softsign_op.cc",
        "//tensorflow/core:kernels/softsign_op.h",
        "//tensorflow/core:kernels/sparse_to_dense_op.cc",
        "//tensorflow/core:kernels/stack_ops.cc",
        "//tensorflow/core:kernels/tile_ops.cc",
        "//tensorflow/core:kernels/tile_ops.h",
        "//tensorflow/core:kernels/topk_op.cc",
        "//tensorflow/core:kernels/training_ops.cc",
        "//tensorflow/core:kernels/training_ops.h",
        "//tensorflow/core:kernels/transpose_op.cc",
        "//tensorflow/core:kernels/transpose_op.h",
        "//tensorflow/core:kernels/transpose_op_functor.h",
        "//tensorflow/core:kernels/where_op.cc",
        "//tensorflow/core:kernels/where_op.h",
        "//tensorflow/core:kernels/xent_op.cc",
        "//tensorflow/core:kernels/xent_op.h",
    ],
    visibility = ["//visibility:public"],
)

# Config settings for determining if we are building for Android
# (the android_32 and android_64 entries are only relevant to internal Google builds).
config_setting(
    name = "android",
    values = {
        "crosstool_top": "//external:android/crosstool",
    },
)

config_setting(
    name = "android_32",
    values = {
        "crosstool_top": "//external:android/crosstool",
        "custom_malloc": "//third_party/java/android/android_ndk_linux/toolchains/gcc_4_8:custom_malloc",
    },
)

config_setting(
    name = "android_64",
    values = {
        "crosstool_top": "//external:android/crosstool",
        "custom_malloc": "//third_party/java/android/android_ndk_linux/toolchains/gcc_4_9:custom_malloc",
    },
)

ANDROID_TF_SOURCES = [
    "//tensorflow/core:android_core_ops",
    "//tensorflow/core:android_extended_ops",
    "//tensorflow/core:android_srcs",
]

ANDROID_TF_COPTS = [
    "-mfpu=neon",
    "-std=c++11",
    "-DMIN_LOG_LEVEL=0",
    "-O2",
]

# Native library support for Android applications.
# Compiles to a trivial library on non-android.
cc_library(
    name = "android_tensorflow_lib",
    srcs = select({
        ":android": ANDROID_TF_SOURCES,
        ":android_32": ANDROID_TF_SOURCES,
        ":android_64": ANDROID_TF_SOURCES,
        "//conditions:default": [],
    }),
    hdrs = [
        "framework/tensor.h",
        "platform/default/integral_types.h",
        "platform/env.h",
        "platform/logging.h",
        "platform/platform.h",
        "platform/types.h",
        "public/session.h",
    ],
    copts = select({
        ":android": ANDROID_TF_COPTS,
        ":android_32": ANDROID_TF_COPTS,
        ":android_64": ANDROID_TF_COPTS,
        "//conditions:default": [],
    }),
    tags = [
        "manual",
        "notap",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@re2//:re2",
        ":protos_cc",
        "//third_party/eigen3",
    ],
)

# -----------------------------------------------------------------------------
# Internal targets

cc_library(
    name = "lib_internal",
    srcs = glob(
        [
            "lib/**/*.h",
            "lib/**/*.cc",
            "platform/*.h",
            "platform/*.cc",
        ] + tf_additional_lib_srcs(),
        exclude = [
            "**/*test*",
            "platform/**/cuda.h",
            "platform/**/stream_executor.h",
        ],
    ),
    hdrs = [
        "lib/core/blocking_counter.h",
        "lib/core/refcount.h",
        "lib/gtl/edit_distance.h",
        "lib/gtl/int_type.h",
        "lib/gtl/iterator_range.h",
        "lib/gtl/top_n.h",
        "lib/io/iterator.h",
        "lib/io/match.h",
        "lib/jpeg/jpeg_handle.h",
        "lib/png/png_io.h",
        "lib/random/random.h",
        "lib/random/random_distributions.h",
        "lib/random/weighted_picker.h",
        "lib/strings/ordered_code.h",
        "lib/strings/regexp.h",
        "platform/platform.h",
        "platform/tensor_coding.h",
        "platform/tracing.h",
    ],
    copts = tf_copts(),
    linkopts = ["-ldl"],
    deps = [
        ":protos_all_cc",
        "//tensorflow/core/platform/default/build_config:platformlib",
    ],
)

tf_cuda_library(
    name = "framework_internal",
    srcs = glob(
        [
            "framework/**/*.h",
            "framework/**/*.cc",
            "public/version.h",
            "util/**/*.h",
            "util/**/*.cc",
        ],
        exclude = [
            "**/*test*",
            "**/*main.cc",
            "framework/fake_input.*",
        ],
    ),
    hdrs = [
        "framework/graph_def_util.h",
        "framework/op_segment.h",
        "framework/rendezvous.h",  # only needed for tests
        "framework/tensor_reference.h",
        "framework/tracking_allocator.h",  # only needed for tests
        "framework/unique_tensor_references.h",
        "util/tensor_slice_set.h",
        "util/tensor_slice_util.h",
    ],
    copts = tf_copts(),
    linkopts = ["-ldl"],
    deps = [
        ":lib",
        ":lib_internal",
        ":protos_all_cc",
        "//third_party/eigen3",
    ],
    alwayslink = 1,
)

filegroup(
    name = "framework_headers",
    srcs = [
        "framework/allocator.h",
        "framework/attr_value_util.h",
        "framework/bfloat16.h",
        "framework/cancellation.h",
        "framework/control_flow.h",
        "framework/device_base.h",
        "framework/function.h",
        "framework/kernel_def_builder.h",
        "framework/node_def_util.h",
        "framework/numeric_types.h",
        "framework/op.h",
        "framework/op_def_builder.h",
        "framework/op_def_util.h",
        "framework/op_kernel.h",
        "framework/partial_tensor_shape.h",
        "framework/rendezvous.h",
        "framework/tensor.h",
        "framework/tensor_reference.h",
        "framework/tensor_shape.h",
        "framework/tensor_types.h",
        "framework/tracking_allocator.h",
        "framework/type_traits.h",
        "framework/types.h",
        "framework/unique_tensor_references.h",
        "lib/core/errors.h",
        "lib/core/notification.h",
        "lib/core/refcount.h",
        "lib/core/status.h",
        "lib/core/stringpiece.h",
        "lib/gtl/array_slice.h",
        "lib/gtl/array_slice_internal.h",
        "lib/gtl/inlined_vector.h",
        "lib/gtl/manual_constructor.h",
        "lib/hash/hash.h",
        "lib/strings/numbers.h",
        "lib/strings/str_util.h",
        "lib/strings/strcat.h",
        "platform/default/dynamic_annotations.h",
        "platform/default/integral_types.h",
        "platform/default/logging.h",
        "platform/default/mutex.h",
        "platform/default/protobuf.h",
        "platform/default/thread_annotations.h",
        "platform/env.h",
        "platform/host_info.h",
        "platform/logging.h",
        "platform/macros.h",
        "platform/mem.h",
        "platform/mutex.h",
        "platform/platform.h",
        "platform/protobuf.h",
        "platform/thread_annotations.h",
        "platform/types.h",
        "public/version.h",
        "util/device_name_utils.h",
    ],
)

tf_cuda_library(
    name = "stream_executor",
    srcs = tf_additional_stream_executor_srcs(),
    hdrs = [
        "platform/cuda.h",
        "platform/stream_executor.h",
    ],
    deps = [
        "//tensorflow/core/platform/default/build_config:stream_executor",
    ],
)

# TODO(josh11b): Is this needed, or can we just use ":protos_all"?
cc_library(
    name = "protos_cc",
    deps = ["//tensorflow/core/platform/default/build_config:protos_cc"],
)

tf_cuda_library(
    name = "core_cpu_internal",
    srcs = glob(
        [
            "client/**/*.cc",
            "common_runtime/*.h",
            "common_runtime/*.cc",
            "framework/versions.h",
            "graph/**/*.h",
            "graph/**/*.cc",
            "public/session.h",
            "public/session_options.h",
            "public/tensor_c_api.h",
            "public/version.h",
        ],
        exclude = [
            "**/*test*",
            "**/*main.cc",
            "common_runtime/gpu_device_factory.cc",
            "common_runtime/direct_session.cc",
            "common_runtime/direct_session.h",
        ],
    ),
    hdrs = glob(
        [
            "common_runtime/*.h",
            "graph/**/*.h",
        ],
        exclude = [
            "**/*test*",
            "common_runtime/direct_session.h",
        ],
    ),
    copts = tf_copts(),
    deps = [
        ":framework",
        ":framework_internal",
        ":lib",
        ":lib_internal",
        ":protos_all_cc",
        "//third_party/eigen3",
    ],
    alwayslink = 1,
)

cc_library(
    name = "cuda",
    deps = [
        "//tensorflow/core/platform/default/build_config:cuda",
    ],
)

tf_cuda_library(
    name = "direct_session",
    srcs = ["common_runtime/direct_session.cc"],
    hdrs = ["common_runtime/direct_session.h"],
    copts = tf_copts(),
    cuda_deps = [
        ":cuda",
    ],
    linkstatic = 1,
    deps = [
        ":core_cpu_internal",
        ":framework",
        ":framework_internal",
        ":lib",
        ":lib_internal",
        ":protos_all_cc",
    ],
    alwayslink = 1,
)

tf_cuda_library(
    name = "gpu_runtime",
    srcs = glob(
        [
            "common_runtime/gpu/*.cc",
        ],
        exclude = [
            "**/*main.cc",
            "**/*test.cc",
        ],
    ),
    hdrs = glob(["common_runtime/gpu/*.h"]),
    copts = tf_copts(),
    cuda_deps = [
        ":cuda",
    ],
    linkstatic = 1,
    deps = [
        ":core_cpu",
        ":core_cpu_internal",
        ":framework",
        ":framework_internal",
        ":lib",
        ":lib_internal",
        ":protos_all_cc",
        ":stream_executor",
        "//third_party/eigen3",
    ],
    alwayslink = 1,
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
        "platform/snappy.h",
    ],
    deps = [":lib_internal"],
)

# Main program for tests
cc_library(
    name = "test_main",
    testonly = 1,
    srcs = ["platform/test_main.cc"],
    copts = tf_copts(),
    linkopts = ["-lm"],
    deps = [
        ":lib",
        ":lib_internal",
        ":test",
        "//tensorflow/core/platform/default/build_config:test_main",
    ],
)

# Low level library tests
tf_cc_tests(
    tests = glob(
        [
            "lib/**/*_test.cc",
            "platform/**/*_test.cc",
        ],
        exclude = ["lib/strings/ordered_code_test.cc"],
    ),
    deps = [
        ":lib",
        ":lib_internal",
        ":lib_test_internal",
        ":protos_all_cc",
        ":test",
        ":test_main",
    ],
)

cc_test(
    name = "lib_jpeg_jpeg_mem_unittest",
    srcs = ["lib/jpeg/jpeg_mem_unittest.cc"],
    data = glob(["lib/jpeg/testdata/*.jpg"]),
    deps = [
        ":lib",
        ":lib_internal",
        ":test",
        ":test_main",
    ],
)

cc_test(
    name = "lib_strings_ordered_code_test",
    srcs = ["lib/strings/ordered_code_test.cc"],
    copts = ["$(STACK_FRAME_UNLIMITED)"],  # Tests initialize large vectors
    deps = [
        ":lib",
        ":lib_internal",
        ":test",
        ":test_main",
    ],
)

# higher level tests
tf_cc_tests(
    linkstatic = tf_kernel_tests_linkstatic(),
    tests = glob(
        [
            "client/**/*_test.cc",
            "common_runtime/**/*_test.cc",
            "framework/**/*_test.cc",
            "graph/**/*_test.cc",
            "util/**/*_test.cc",
            "ops/**/*_test.cc",
        ],
        exclude = [
            # TODO(opensource): fix
            "common_runtime/gpu/*_test.cc",
            # Run by tests below
            "common_runtime/gpu/gpu_allocator_retry_test.cc",
            "common_runtime/gpu/gpu_bfc_allocator_test.cc",
            "common_runtime/gpu/gpu_region_allocator_test.cc",
        ],
    ),
    deps = [
        ":all_kernels",
        ":core",
        ":core_cpu",
        ":core_cpu_internal",
        ":direct_session",
        ":framework",
        ":framework_internal",
        ":kernel_lib",
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

# GPU-related tests
tf_cc_tests(
    linkstatic = tf_kernel_tests_linkstatic(),
    tags = tf_cuda_tests_tags(),
    tests = glob(
        [
            "kernels/**/*_test.cc",
            "user_ops/**/*_test.cc",
            "common_runtime/gpu/*_test.cc",
        ],
        exclude = [
            # Run by tests below
            "common_runtime/gpu/gpu_allocator_retry_test.cc",
        ],
    ),
    deps = [
        ":all_kernels",
        ":core_cpu",
        ":core_cpu_internal",
        ":direct_session",
        ":framework",
        ":framework_internal",
        ":gpu_runtime",
        ":kernel_lib",
        ":lib",
        ":lib_internal",
        ":protos_all_cc",
        ":test",
        ":test_main",
        ":testlib",
        "//tensorflow/cc:cc_ops",
    ],
)

tf_cc_tests(
    linkstatic = tf_kernel_tests_linkstatic(),
    tags = tf_cuda_tests_tags() + ["nomac"],
    tests = ["common_runtime/gpu/gpu_allocator_retry_test.cc"],
    deps = [
        ":all_kernels",
        ":core_cpu",
        ":core_cpu_internal",
        ":direct_session",
        ":framework",
        ":framework_internal",
        ":gpu_runtime",
        ":kernel_lib",
        ":lib",
        ":lib_internal",
        ":protos_all_cc",
        ":test",
        ":test_main",
        ":testlib",
        "//tensorflow/cc:cc_ops",
    ],
)

# Test data
filegroup(
    name = "image_testdata",
    srcs = [
        # PNG data
        "lib/png/testdata/lena_gray.png",
        "lib/png/testdata/lena_rgba.png",
        # JPEG data
        "lib/jpeg/testdata/jpeg_merge_test1.jpg",
        "lib/jpeg/testdata/jpeg_merge_test1_cmyk.jpg",
        # Corrupted JPEG files for tests
        "lib/jpeg/testdata/bad_huffman.jpg",
        "lib/jpeg/testdata/corrupt.jpg",
        # -- hand-edited variant: stops at line 0
        "lib/jpeg/testdata/corrupt34_2.jpg",
        # -- hand-edited variant: stops at line 4
        "lib/jpeg/testdata/corrupt34_3.jpg",
        # -- hand-edited variant: stops after a restart marker
        "lib/jpeg/testdata/corrupt34_4.jpg",
    ],
)

# -----------------------------------------------------------------------------
# Google-internal targets go here (must be at the end).

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)

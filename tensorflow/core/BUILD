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
# ":all_kernels" - The cpu-specific kernels, plus gpu kernels if
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
# cc_library ":android_tensorflow_lib" - Native library
# portable_proto_library ":android_proto_lib" (Google-internal)

package(default_visibility = ["//tensorflow:internal"])

licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "tf_copts")
load("//tensorflow:tensorflow.bzl", "tf_cc_tests")
load("//tensorflow:tensorflow.bzl", "tf_cuda_library")
load("//tensorflow:tensorflow.bzl", "tf_gen_op_libs")

# For platform specific build config
load(
    "//tensorflow/core:platform/default/build_config.bzl",
    "tf_proto_library",
    "tf_proto_library_cc",
    "tf_additional_lib_srcs",
    "tf_additional_stream_executor_srcs",
    "tf_additional_test_deps",
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
    srcs = glob(
        ["**/*.proto"],
        exclude = [
            "protobuf/worker.proto",
            "protobuf/worker_service.proto",
            "protobuf/master.proto",
            "protobuf/master_service.proto",
        ],
    ),
    cc_api_version = 2,
    go_api_version = 2,
    java_api_version = 2,
    py_api_version = 2,
    visibility = ["//visibility:public"],
)

tf_proto_library_cc(
    name = "worker_proto",
    srcs = ["protobuf/worker.proto"],
    cc_api_version = 2,
    cc_libs = [":protos_all_cc"],
    visibility = [
        "//tensorflow:internal",
    ],
)

tf_proto_library_cc(
    name = "worker_service_proto",
    srcs = ["protobuf/worker_service.proto"],
    has_services = 1,
    cc_api_version = 2,
    cc_grpc_version = 1,
    cc_libs = [":worker_proto_cc"],
    cc_stubby_versions = ["2"],
    visibility = [
        "//tensorflow:internal",
    ],
)

tf_proto_library_cc(
    name = "master_proto",
    srcs = ["protobuf/master.proto"],
    cc_api_version = 2,
    cc_libs = [":protos_all_cc"],
    visibility = [
        "//tensorflow:internal",
    ],
)

tf_proto_library_cc(
    name = "master_service_proto",
    srcs = ["protobuf/master_service.proto"],
    has_services = 1,
    cc_api_version = 2,
    cc_grpc_version = 1,
    cc_libs = [":master_proto_cc"],
    cc_stubby_versions = ["2"],
    visibility = [
        "//tensorflow:internal",
    ],
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
        "util/reporter.cc",
    ] + tf_additional_test_srcs(),
    hdrs = [
        "lib/core/status_test_util.h",
        "platform/test.h",
        "platform/test_benchmark.h",
        "util/reporter.h",
    ],
    copts = tf_copts(),
    linkopts = ["-lm"],
    visibility = ["//visibility:public"],
    deps = [
        ":lib",
        ":lib_internal",
        "//tensorflow/core/platform/default/build_config:gtest",
    ] + tf_additional_test_deps(),
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
        "framework/graph_def_util.h",
        "framework/kernel_def_builder.h",
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
        "framework/tracking_allocator.h",
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
        "util/tensor_format.h",
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

# Generates library per group of ops.
tf_gen_op_libs(
    op_lib_names = [
        "array_ops",
        "candidate_sampling_ops",
        "control_flow_ops",
        "ctc_ops",
        "data_flow_ops",
        "function_ops",
        "functional_ops",
        "image_ops",
        "io_ops",
        "linalg_ops",
        "logging_ops",
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

cc_library(
    name = "ops",
    visibility = ["//visibility:public"],
    deps = [
        ":array_ops_op_lib",
        ":candidate_sampling_ops_op_lib",
        ":control_flow_ops_op_lib",
        ":ctc_ops_op_lib",
        ":data_flow_ops_op_lib",
        ":function_ops_op_lib",
        ":functional_ops_op_lib",
        ":image_ops_op_lib",
        ":io_ops_op_lib",
        ":linalg_ops_op_lib",
        ":logging_ops_op_lib",
        ":math_ops_op_lib",
        ":nn_ops_op_lib",
        ":no_op_op_lib",
        ":parsing_ops_op_lib",
        ":random_ops_op_lib",
        ":script_ops_op_lib",
        ":sendrecv_ops_op_lib",
        ":sparse_ops_op_lib",
        ":state_ops_op_lib",
        ":string_ops_op_lib",
        ":training_ops_op_lib",
        ":user_ops_op_lib",
        "//tensorflow/models/embedding:word2vec_ops",
    ],
    alwayslink = 1,
)

cc_library(
    name = "array_grad",
    srcs = ["ops/array_grad.cc"],
    linkstatic = 1,  # Seems to be needed since alwayslink is broken in bazel
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
    linkstatic = 1,  # Seems to be needed since alwayslink is broken in bazel
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
    srcs = ["ops/math_grad.cc"],
    linkstatic = 1,  # Seems to be needed since alwayslink is broken in bazel
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
    linkstatic = 1,  # Seems to be needed since alwayslink is broken in bazel
    visibility = ["//visibility:public"],
    deps = [
        ":framework",
        ":lib",
        ":nn_ops_op_lib",
    ],
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

# This includes implementations of all kernels built into TensorFlow.
cc_library(
    name = "all_kernels",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core/kernels:array",
        "//tensorflow/core/kernels:candidate_sampler_ops",
        "//tensorflow/core/kernels:control_flow_ops",
        "//tensorflow/core/kernels:ctc_ops",
        "//tensorflow/core/kernels:data_flow",
        "//tensorflow/core/kernels:fact_op",
        "//tensorflow/core/kernels:image",
        "//tensorflow/core/kernels:io",
        "//tensorflow/core/kernels:linalg",
        "//tensorflow/core/kernels:logging",
        "//tensorflow/core/kernels:math",
        "//tensorflow/core/kernels:nn",
        "//tensorflow/core/kernels:parsing",
        "//tensorflow/core/kernels:random_ops",
        "//tensorflow/core/kernels:required",
        "//tensorflow/core/kernels:sparse",
        "//tensorflow/core/kernels:state",
        "//tensorflow/core/kernels:string",
        "//tensorflow/core/kernels:training_ops",
        "//tensorflow/models/embedding:word2vec_kernels",
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
        "graph/testlib.cc",
    ],
    hdrs = [
        "common_runtime/kernel_benchmark_testlib.h",
        "framework/fake_input.h",
        "framework/function_testlib.h",
        "framework/tensor_testutil.h",
        "graph/testlib.h",
        # TODO(josh11b): Drop this once users are depending on
        # kernels:ops_testutil instead.
        "//tensorflow/core/kernels:ops_testutil.h",
    ],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":all_kernels",
        ":core_cpu",
        ":core_cpu_internal",
        ":framework",
        ":framework_internal",
        ":lib",
        ":protos_all_cc",
        ":tensor_testutil",
        ":tensorflow",  # TODO(josh11b): Drop this dep?
        ":test",
        "//tensorflow/core/kernels:constant_op",
        "//tensorflow/core/kernels:ops_testutil",
        "//tensorflow/core/kernels:ops_util",
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
    srcs = [
        "//tensorflow/core/kernels:android_srcs",
        "//tensorflow/core/util/ctc:android_srcs",
    ] + glob(
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
            "util/reporter.cc",
            "util/reporter.h",
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
            "platform/test_benchmark.h",
            "platform/default/test_benchmark.cc",
            "user_ops/**/*.cu.cc",
            "common_runtime/gpu/*.cc",
            "common_runtime/gpu_device_factory.cc",
        ],
    ),
    visibility = ["//visibility:public"],
)

# Config setting for determining if we are building for Android.
config_setting(
    name = "android",
    values = {
        "crosstool_top": "//external:android/crosstool",
    },
)

ANDROID_TF_COPTS = [
    "-mfpu=neon",
    "-std=c++11",
    "-DMIN_LOG_LEVEL=0",
    "-O2",
]

# Native library support for Android applications.
# Does not contain operators, use :android_tensorflow_lib if you want full
# operator support.
# Compiles to a trivial library on non-android to prevent irrelevant
# build errors.
cc_library(
    name = "android_tensorflow_lib_lite",
    srcs = select({
        ":android": ["//tensorflow/core:android_srcs"],
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

# Full Tensorflow library with operator support. Use this unless reducing
# binary size (by packaging a reduced operator set) is a concern.
cc_library(
    name = "android_tensorflow_lib",
    srcs = [
        "//tensorflow/core/kernels:android_core_ops",
        "//tensorflow/core/kernels:android_extended_ops",
    ],
    copts = select({
        ":android": ANDROID_TF_COPTS,
        "//conditions:default": [],
    }),
    tags = [
        "manual",
        "notap",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@re2//:re2",
        ":android_tensorflow_lib_lite",
        ":protos_cc",
        "//third_party/eigen3",
    ],
)

# -----------------------------------------------------------------------------
# Libraries for GPU facilities that are useful for writing kernels.
cc_library(
    name = "gpu_lib",
    srcs = [
        "common_runtime/gpu/gpu_event_mgr.cc",
    ],
    hdrs = [
        "common_runtime/gpu/gpu_event_mgr.h",
        "common_runtime/gpu_device_context.h",
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
        "lib/strings/scanner.h",
        "platform/denormal.h",
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
            "util/reporter.h",
            "util/reporter.cc",
            "framework/fake_input.*",
        ],
    ),
    hdrs = [
        "framework/op_segment.h",
        "framework/rendezvous.h",  # only needed for tests
        "framework/tensor_reference.h",
        "framework/tracking_allocator.h",  # only needed for tests
        "framework/unique_tensor_references.h",
        "util/command_line_flags.h",
        "util/tensor_slice_set.h",
        "util/tensor_slice_util.h",
    ],
    copts = tf_copts(),
    linkopts = [
        "-ldl",
        "-lm",
    ],
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
            "common_runtime/direct_session.cc",
            "common_runtime/direct_session.h",
            "common_runtime/gpu_device_context.h",
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
            "common_runtime/gpu_device_context.h",
        ],
    ),
    copts = tf_copts(),
    deps = [
        ":framework",
        ":framework_internal",
        ":function_ops_op_lib",
        ":functional_grad",
        ":functional_ops_op_lib",
        ":lib",
        ":lib_internal",
        ":protos_all_cc",
        "//tensorflow/core/kernels:required",
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
        ":lib",
        ":lib_internal",
        ":protos_all_cc",
    ],
    alwayslink = 1,
)

tf_cuda_library(
    name = "gpu_runtime",
    srcs = [
        "common_runtime/gpu/gpu_allocator_retry.cc",
        "common_runtime/gpu/gpu_bfc_allocator.cc",
        "common_runtime/gpu/gpu_debug_allocator.cc",
        "common_runtime/gpu/gpu_device.cc",
        "common_runtime/gpu/gpu_device_factory.cc",
        "common_runtime/gpu/gpu_init.cc",
        "common_runtime/gpu/gpu_stream_util.cc",
        "common_runtime/gpu/gpu_util.cc",
        "common_runtime/gpu/gpu_util_platform_specific.cc",
        "common_runtime/gpu/pool_allocator.cc",
        "common_runtime/gpu/process_state.cc",
    ],
    hdrs = [
        "common_runtime/gpu/gpu_allocator_retry.h",
        "common_runtime/gpu/gpu_bfc_allocator.h",
        "common_runtime/gpu/gpu_debug_allocator.h",
        "common_runtime/gpu/gpu_device.h",
        "common_runtime/gpu/gpu_init.h",
        "common_runtime/gpu/gpu_stream_util.h",
        "common_runtime/gpu/gpu_util.h",
        "common_runtime/gpu/pool_allocator.h",
        "common_runtime/gpu/process_state.h",
        "common_runtime/gpu/visitable_allocator.h",
    ],
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
        ":gpu_lib",
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
        ":lib",
        ":lib_internal",
        ":ops",
        ":protos_all_cc",
        ":test",
        ":test_main",
        ":testlib",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core/kernels:ops_util",
        "//third_party/eigen3",
    ],
)

# GPU-related tests
tf_cc_tests(
    linkstatic = tf_kernel_tests_linkstatic(),
    tags = tf_cuda_tests_tags(),
    tests = glob(
        [
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

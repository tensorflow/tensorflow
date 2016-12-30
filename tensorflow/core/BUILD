# Description:
# TensorFlow is a computational framework, primarily for use in machine
# learning applications.
#
# Public targets:
#
# ":protos_all" - exports all core TensorFlow protos
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
# cc_library ":android_tensorflow_lib_selective_registration" - Native library
#   supporting SELECTIVE_REGISTRATION feature.
# portable_proto_library ":android_proto_lib" (Google-internal)

package(default_visibility = [
    "//tensorflow:internal",
    "//tensorflow_models:__subpackages__",
])

licenses(["notice"])  # Apache 2.0

load(
    "//tensorflow:tensorflow.bzl",
    "if_android",
    "if_ios",
    "if_not_windows",
    "tf_copts",
    "tf_cc_test",
    "tf_cc_tests",
    "tf_cuda_library",
    "tf_gen_op_libs",
    "tf_generate_proto_text_sources",
    "tf_genrule_cmd_append_to_srcs",
    "tf_opts_nortti_if_android",
    "cc_header_only_library",
)
load("//tensorflow:tensorflow.bzl", "tf_cc_test_gpu")
load("//tensorflow:tensorflow.bzl", "tf_cc_tests_gpu")
load("//tensorflow:tensorflow.bzl", "tf_version_info_genrule")

# For platform specific build config
load(
    "//tensorflow/core:platform/default/build_config.bzl",
    "tf_proto_library",
    "tf_proto_library_cc",
    "tf_additional_lib_hdrs",
    "tf_additional_lib_srcs",
    "tf_additional_minimal_lib_srcs",
    "tf_additional_proto_hdrs",
    "tf_additional_proto_srcs",
    "tf_additional_lib_deps",
    "tf_additional_stream_executor_srcs",
    "tf_additional_cupti_wrapper_deps",
    "tf_additional_libdevice_data",
    "tf_additional_libdevice_deps",
    "tf_additional_libdevice_srcs",
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

# Protos which are needed for core tensorflow, including on mobile builds.
#
# Note that some protos are in neither additional_core_proto_srcs nor this
# filegroup; e.g.  ones with individual proto_library targets.
CORE_PROTO_SRCS = [
    "example/example.proto",
    "example/feature.proto",
    "framework/allocation_description.proto",
    "framework/attr_value.proto",
    "framework/cost_graph.proto",
    "framework/device_attributes.proto",
    "framework/function.proto",
    "framework/graph.proto",
    "framework/kernel_def.proto",
    "framework/log_memory.proto",
    "framework/node_def.proto",
    "framework/op_def.proto",
    "framework/resource_handle.proto",
    "framework/step_stats.proto",
    "framework/summary.proto",
    "framework/tensor.proto",
    "framework/tensor_description.proto",
    "framework/tensor_shape.proto",
    "framework/tensor_slice.proto",
    "framework/types.proto",
    "framework/versions.proto",
    "lib/core/error_codes.proto",
    "protobuf/config.proto",
    "protobuf/debug.proto",
    "protobuf/tensor_bundle.proto",
    "protobuf/saver.proto",
    "util/memmapped_file_system.proto",
    "util/saved_tensor_slice.proto",
]

# Protos which are not needed on mobile builds, but should be included in
# protos_all.
#
# Note that some protos are in neither core_proto_srcs nor this filegroup; e.g.
# ones with individual proto_library targets.
ADDITIONAL_CORE_PROTO_SRCS = [
    "example/example_parser_configuration.proto",
    "framework/variable.proto",
    "protobuf/control_flow.proto",
    "protobuf/meta_graph.proto",
    "protobuf/named_tensor.proto",
    "protobuf/queue_runner.proto",
    "protobuf/saved_model.proto",
    "protobuf/tensorflow_server.proto",
    "util/event.proto",
    "util/test_log.proto",
]

tf_proto_library(
    name = "protos_all",
    srcs = CORE_PROTO_SRCS + ADDITIONAL_CORE_PROTO_SRCS,
    cc_api_version = 2,
    go_api_version = 2,
    java_api_version = 2,
    js_api_version = 2,
    js_codegen = "jspb",
    visibility = ["//visibility:public"],
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
    ] + glob(tf_additional_proto_hdrs()),
    copts = tf_copts(),
    deps = [
        ":protos_all_cc",
        "//tensorflow/core/platform/default/build_config:proto_parsing",
    ],
)

cc_library(
    name = "lib",
    hdrs = [
        "lib/core/arena.h",
        "lib/core/bitmap.h",
        "lib/core/bits.h",
        "lib/core/casts.h",
        "lib/core/coding.h",
        "lib/core/errors.h",
        "lib/core/notification.h",
        "lib/core/raw_coding.h",
        "lib/core/status.h",
        "lib/core/stringpiece.h",
        "lib/core/threadpool.h",
        "lib/gtl/array_slice.h",
        "lib/gtl/cleanup.h",
        "lib/gtl/flatmap.h",
        "lib/gtl/flatset.h",
        "lib/gtl/inlined_vector.h",
        "lib/gtl/priority_queue_util.h",
        "lib/hash/crc32c.h",
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
        "lib/monitoring/counter.h",
        "lib/monitoring/sampler.h",
        "lib/random/distribution_sampler.h",
        "lib/random/philox_random.h",
        "lib/random/simple_philox.h",
        "lib/strings/numbers.h",
        "lib/strings/str_util.h",
        "lib/strings/strcat.h",
        "lib/strings/stringprintf.h",
        "platform/cpu_info.h",
        "platform/dynamic_annotations.h",
        "platform/env.h",
        "platform/file_system.h",
        "platform/fingerprint.h",
        "platform/init_main.h",
        "platform/logging.h",
        "platform/macros.h",
        "platform/mem.h",
        "platform/mutex.h",
        "platform/net.h",
        "platform/notification.h",
        "platform/prefetch.h",
        "platform/profile_utils/cpu_utils.h",
        "platform/protobuf.h",
        "platform/stacktrace.h",
        "platform/strong_hash.h",
        "platform/subprocess.h",
        "platform/thread_annotations.h",
        "platform/types.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":lib_internal",
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
        "example/feature_util.h",
        "framework/allocator.h",
        "framework/attr_value_util.h",
        "framework/bfloat16.h",
        "framework/cancellation.h",
        "framework/common_shape_fns.h",
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
        "framework/resource_op_kernel.h",
        "framework/selective_registration.h",
        "framework/session_state.h",
        "framework/shape_inference.h",
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
        "util/example_proto_fast_parsing.h",
        "util/example_proto_helper.h",
        "util/guarded_philox_random.h",
        "util/mirror_pad_mode.h",
        "util/padding.h",
        "util/port.h",
        "util/saved_tensor_slice_util.h",
        "util/sparse/group_iterator.h",
        "util/sparse/sparse_tensor.h",
        "util/stat_summarizer.h",
        "util/stream_executor_util.h",
        "util/strided_slice_op.h",
        "util/tensor_format.h",
        "util/tensor_slice_reader.h",
        "util/tensor_slice_reader_cache.h",
        "util/tensor_slice_writer.h",
        "util/use_cudnn.h",
        "util/util.h",
        "util/work_sharder.h",
    ] + select({
        "//tensorflow:windows": [],
        "//conditions:default": [
            "util/memmapped_file_system.h",
            "util/memmapped_file_system_writer.h",
        ],
    }),
    visibility = ["//visibility:public"],
    deps = [":framework_internal"],
)

cc_library(
    name = "framework_lite",
    srcs = tf_additional_minimal_lib_srcs(),
    hdrs = [
        "framework/numeric_types.h",
        "framework/tensor_types.h",
        "framework/type_traits.h",
        "platform/default/dynamic_annotations.h",
        "platform/default/integral_types.h",
        "platform/default/mutex.h",
        "platform/default/thread_annotations.h",
        "platform/dynamic_annotations.h",
        "platform/macros.h",
        "platform/mutex.h",
        "platform/platform.h",
        "platform/prefetch.h",
        "platform/thread_annotations.h",
        "platform/types.h",
    ],
    visibility = ["//visibility:public"],
    deps =
        [
            "//tensorflow/core/platform/default/build_config:minimal",
            "//third_party/eigen3",
        ],
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
        "resource_variable_ops",
        "sdca_ops",
        "set_ops",
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
    name = "word2vec_ops",
    srcs = ["ops/word2vec_ops.cc"],
    linkstatic = 1,
    visibility = ["//tensorflow:internal"],
    deps = ["//tensorflow/core:framework"],
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
        ":sdca_ops_op_lib",
        ":sendrecv_ops_op_lib",
        ":set_ops_op_lib",
        ":sparse_ops_op_lib",
        ":state_ops_op_lib",
        ":string_ops_op_lib",
        ":training_ops_op_lib",
        ":user_ops_op_lib",
        ":word2vec_ops",
    ],
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
    ],
    alwayslink = 1,
)

tf_cuda_library(
    name = "core_cpu",
    hdrs = [
        "common_runtime/device.h",
        "common_runtime/shape_refiner.h",
        "graph/algorithm.h",
        "graph/default_device.h",
        "graph/gradients.h",
        "graph/graph.h",
        "graph/graph_constructor.h",
        "graph/graph_def_builder.h",
        "graph/node_builder.h",
        "graph/validate.h",
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
    name = "all_kernels",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core/kernels:array",
        "//tensorflow/core/kernels:candidate_sampler_ops",
        "//tensorflow/core/kernels:control_flow_ops",
        "//tensorflow/core/kernels:ctc_ops",
        "//tensorflow/core/kernels:data_flow",
        "//tensorflow/core/kernels:fake_quant_ops",
        "//tensorflow/core/kernels:function_ops",
        "//tensorflow/core/kernels:image",
        "//tensorflow/core/kernels:io",
        "//tensorflow/core/kernels:linalg",
        "//tensorflow/core/kernels:logging",
        "//tensorflow/core/kernels:math",
        "//tensorflow/core/kernels:multinomial_op",
        "//tensorflow/core/kernels:nn",
        "//tensorflow/core/kernels:parameterized_truncated_normal_op",
        "//tensorflow/core/kernels:parsing",
        "//tensorflow/core/kernels:random_ops",
        "//tensorflow/core/kernels:required",
        "//tensorflow/core/kernels:resource_variable_ops",
        "//tensorflow/core/kernels:sdca_ops",
        "//tensorflow/core/kernels:set_kernels",
        "//tensorflow/core/kernels:sparse",
        "//tensorflow/core/kernels:state",
        "//tensorflow/core/kernels:string",
        "//tensorflow/core/kernels:training_ops",
        "//tensorflow/core/kernels:word2vec_kernels",
    ] + if_not_windows([
        "//tensorflow/core/kernels:fact_op",
        "//tensorflow/core/kernels:array_not_windows",
        "//tensorflow/core/kernels:math_not_windows",
        "//tensorflow/core/kernels:quantized_ops",
    ]),
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
        ":framework",
        ":framework_internal",
        ":lib",
        ":lib_internal",
        ":proto_text",
        ":protos_all_cc",
        ":shape_inference_testutil",
        ":tensor_testutil",
        ":test",
        "//tensorflow/core/kernels:constant_op",
        "//tensorflow/core/kernels:ops_testutil",
        "//tensorflow/core/kernels:ops_util",
        "//tensorflow/core/platform/default/build_config:gtest",
    ],
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

# Core sources. Should eventually become identical to open source
# sources.
filegroup(
    name = "android_srcs",
    srcs = [
        ":proto_text_srcs_all",
        "//tensorflow/core/kernels:android_srcs",
        "//tensorflow/core/platform/default/build_config:android_srcs",
        "//tensorflow/core/util/ctc:android_srcs",
        "//tensorflow/core/util/tensor_bundle:android_srcs",
        "common_runtime/gpu/gpu_tracer.cc",
        "common_runtime/gpu/gpu_tracer.h",
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
            "graph/dot.*",
            "lib/jpeg/**/*",
            "lib/png/**/*",
            "lib/gif/**/*",
            "util/events_writer.*",
            "util/reporter.*",
            "platform/default/stream_executor.*",
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
            "common_runtime/gpu/**/*",
            "common_runtime/gpu_device_factory.*",
        ],
    ),
    visibility = ["//visibility:public"],
)

# Native library support for Android applications.  Does not contain
# operators, use :android_tensorflow_lib if you want full operator
# support.
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
    copts = tf_copts() + [
        "-Os",
    ],
    linkopts = ["-lz"],
    tags = [
        "manual",
        "notap",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":protos_cc",
        "//third_party/eigen3",
    ],
    alwayslink = 1,
)

# Native library support for iOS applications.
#
# bazel  build --config=ios_x86_64 \
# //third_party/tensorflow/core:ios_tensorflow_lib
cc_library(
    name = "ios_tensorflow_lib",
    srcs = if_ios([
        ":android_op_registrations_and_gradients",
        "//tensorflow/core:android_srcs",
        "//tensorflow/core/kernels:android_core_ops",
        "//tensorflow/core/kernels:android_extended_ops",
    ]),
    copts = tf_copts() + ["-Os"] + ["-std=c++11"],
    visibility = ["//visibility:public"],
    deps = [
        ":protos_cc",
        "//third_party/eigen3",
        "@gemmlowp//:gemmlowp",
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
        ":protos_cc",
        "//tensorflow/core/kernels:android_tensorflow_kernels",
        "//third_party/eigen3",
    ],
    alwayslink = 1,
)

# Android library for use with the SELECTIVE_REGISTRATION feature.
# Does not contain operators. In contrast to android_tensorflow_lib_lite,
# this links in framework support for all types, relying on selective
# registration of ops to prune code size.
cc_library(
    name = "android_tensorflow_lib_selective_registration",
    srcs = if_android(["//tensorflow/core:android_srcs"]),
    copts = tf_copts() + [
        "-Os",
        "-DSUPPORT_SELECTIVE_REGISTRATION",
    ],
    tags = [
        "manual",
        "notap",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":protos_cc",
        "//third_party/eigen3",
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
    copts = tf_copts() + ["-Os"],
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
        ":proto_text",
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
# Internal targets

tf_proto_library_cc(
    name = "worker_proto",
    srcs = ["protobuf/worker.proto"],
    cc_api_version = 2,
    protodeps = [":protos_all"],
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
    protodeps = [":protos_all"],
    visibility = [
        "//tensorflow:internal",
    ],
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

cc_library(
    name = "lib_internal",
    srcs = select({
        "//tensorflow:windows": glob(
            [
                "lib/**/*.h",
                "lib/**/*.cc",
                "platform/*.h",
                "platform/*.cc",
            ],
            exclude = [
                "**/*test*",
                "lib/gif/**/*",
                "lib/jpeg/**/*",
                "platform/gif.h",
                "platform/jpeg.h",
                "platform/**/cuda.h",
                "platform/**/cuda_libdevice_path.cc",
                "platform/**/stream_executor.h",
                "platform/load_library.cc",
            ],
        ),
        "//conditions:default": glob(
            [
                "lib/**/*.h",
                "lib/**/*.cc",
                "platform/*.h",
                "platform/*.cc",
                "platform/profile_utils/**/*.h",
                "platform/profile_utils/**/*.cc",
            ],
            exclude = [
                "**/*test*",
                "lib/gif/**/*",
                "lib/jpeg/**/*",
                "platform/gif.h",
                "platform/jpeg.h",
                "platform/**/cuda.h",
                "platform/**/cuda_libdevice_path.cc",
                "platform/**/stream_executor.h",
            ],
        ),
    }) + tf_additional_lib_srcs(
        exclude = [
            "**/*test*",
            "platform/**/cuda.h",
            "platform/**/cuda_libdevice_path.cc",
            "platform/**/stream_executor.h",
        ] +
        # Protobuf deps already included through the ":lib_proto_parsing"
        # dependency.
        tf_additional_proto_srcs(),
    ),
    hdrs = tf_additional_lib_hdrs() + [
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
        "lib/monitoring/collected_metrics.h",
        "lib/monitoring/collection_registry.h",
        "lib/monitoring/metric_def.h",
        "lib/monitoring/mobile_counter.h",
        "lib/monitoring/mobile_sampler.h",
        "lib/png/png_io.h",
        "lib/random/random.h",
        "lib/random/random_distributions.h",
        "lib/random/weighted_picker.h",
        "lib/strings/base64.h",
        "lib/strings/ordered_code.h",
        "lib/strings/proto_text_util.h",
        "lib/strings/scanner.h",
        "lib/wav/wav_io.h",
        "platform/demangle.h",
        "platform/denormal.h",
        "platform/host_info.h",
        "platform/platform.h",
        "platform/protobuf_internal.h",
        "platform/tensor_coding.h",
        "platform/tracing.h",
    ],
    copts = tf_copts(),
    linkopts = ["-ldl"],
    deps = [
        ":lib_proto_parsing",
        ":protos_all_cc",
        "//tensorflow/core/platform/default/build_config:platformlib",
        "//third_party/eigen3",
        "@zlib_archive//:zlib",
    ],
)

cc_library(
    name = "gif_internal",
    srcs = [
        "lib/gif/gif_io.cc",
        "platform/gif.h",
    ],
    hdrs = ["lib/gif/gif_io.h"],
    copts = tf_copts(),
    linkopts = ["-ldl"],
    deps = [
        ":lib",
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
    linkopts = ["-ldl"],
    deps = [
        ":lib",
        "//tensorflow/core/platform/default/build_config:jpeg",
    ],
)

proto_text_hdrs_and_srcs = tf_generate_proto_text_sources(
    name = "proto_text_srcs_all",
    srcs = CORE_PROTO_SRCS,
    srcs_relative_dir = "tensorflow/core/",
)

cc_library(
    name = "proto_text",
    srcs = proto_text_hdrs_and_srcs.srcs,
    hdrs = proto_text_hdrs_and_srcs.hdrs,
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

tf_cuda_library(
    name = "framework_internal",
    srcs = glob(
        [
            "example/**/*.h",
            "example/**/*.cc",
            "framework/**/*.h",
            "framework/**/*.cc",
            "util/**/*.h",
            "util/**/*.cc",
        ],
        exclude = [
            "**/*test*",
            "**/*main.cc",
            "example/example_parser_configuration.*",
            "util/reporter.h",
            "util/reporter.cc",
            "framework/fake_input.*",
            "util/memmapped_file_system.*",
            "util/memmapped_file_system_writer.*",
            "util/version_info.cc",
        ],
    ) + select({
        "//tensorflow:windows": [],
        "//conditions:default": [
            "util/memmapped_file_system.h",
            "util/memmapped_file_system.cc",
            "util/memmapped_file_system_writer.h",
            "util/memmapped_file_system_writer.cc",
        ],
    }),
    hdrs = [
        "framework/op_segment.h",
        "framework/rendezvous.h",  # only needed for tests
        "framework/tensor_reference.h",
        "framework/tracking_allocator.h",  # only needed for tests
        "framework/unique_tensor_references.h",
        "util/command_line_flags.h",
        "util/env_var.h",
        "util/presized_cuckoo_map.h",
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
        ":proto_text",
        ":protos_all_cc",
        ":version_lib",
        "//tensorflow/core/kernels:bounds_check",
        "//third_party/eigen3",
    ],
    alwayslink = 1,
)

cc_header_only_library(
    name = "framework_headers_lib",
    visibility = ["//visibility:public"],
    deps = [
        ":framework",
    ],
)

cc_header_only_library(
    name = "stream_executor_headers_lib",
    visibility = ["//visibility:public"],
    deps = [
        ":stream_executor",
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

# Like stream_executor library, but compiles without --config=cuda
# and does not include any cuda dependencies.
cc_library(
    name = "stream_executor_no_cuda",
    srcs = tf_additional_stream_executor_srcs(),
    hdrs = [
        "platform/stream_executor_no_cuda.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core/platform/default/build_config:stream_executor_no_cuda",
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
            "framework/versions.h",
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
        ":proto_text",
        ":protos_all_cc",
        "//third_party/eigen3",
        "//tensorflow/core/kernels:required",
    ] + tf_additional_lib_deps(),
    alwayslink = 1,
)

# This library is deprecated and no longer publicly available.
# Do not add more uses of it.
cc_library(
    name = "regexp_internal",
    hdrs = [
        "platform/regexp.h",
    ],
    visibility = [
        "//tensorflow/compiler:__subpackages__",
        "//tensorflow/tools/tfprof:__subpackages__",
    ],
    deps = [":lib_internal"],
)

tf_cuda_library(
    name = "direct_session_internal",
    srcs = ["common_runtime/direct_session.cc"],
    hdrs = ["common_runtime/direct_session.h"],
    copts = tf_copts(),
    cuda_deps = [
        ":gpu_tracer",
    ],
    linkstatic = 1,
    deps = [
        ":core_cpu_internal",
        ":framework",
        ":lib",
        ":lib_internal",
        ":proto_text",
        ":protos_all_cc",
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
    name = "gpu_tracer",
    srcs = [
        "common_runtime/gpu/gpu_tracer.cc",
    ],
    hdrs = [
        "common_runtime/gpu/gpu_tracer.h",
    ],
    copts = tf_copts(),
    cuda_deps = tf_additional_cupti_wrapper_deps(),
    deps = [
        ":core_cpu_internal",
        ":lib",
        ":protos_all_cc",
    ],
)

tf_cuda_library(
    name = "gpu_runtime",
    srcs = [
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
        "common_runtime/gpu_device_context.h",
    ],
    hdrs = [
        "common_runtime/gpu/gpu_bfc_allocator.h",
        "common_runtime/gpu/gpu_debug_allocator.h",
        "common_runtime/gpu/gpu_device.h",
        "common_runtime/gpu/gpu_init.h",
        "common_runtime/gpu/gpu_stream_util.h",
        "common_runtime/gpu/gpu_util.h",
        "common_runtime/gpu/pool_allocator.h",
        "common_runtime/gpu/process_state.h",
    ],
    copts = tf_copts(),
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
        "common_runtime/sycl/sycl_device_context.h",
    ]),
    copts = tf_copts(),
    linkstatic = 1,
    deps = [
        ":core_cpu",
        ":core_cpu_internal",
        ":framework",
        ":framework_internal",
        ":lib",
        ":lib_internal",
        ":protos_all_cc",
        "//third_party/eigen3",
        "@local_config_sycl//sycl:sycl",
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
    visibility = ["//tensorflow:internal"],
    deps = [
        ":lib",
        ":lib_internal",
        ":test",
        "//tensorflow/core/platform/default/build_config:test_main",
    ],
    alwayslink = 1,
)

tf_cc_tests(
    name = "low_level_library_tests",
    size = "small",
    srcs = [
        "lib/core/arena_test.cc",
        "lib/core/bit_cast_test.cc",
        "lib/core/bitmap_test.cc",
        "lib/core/blocking_counter_test.cc",
        "lib/core/coding_test.cc",
        "lib/core/notification_test.cc",
        "lib/core/refcount_test.cc",
        "lib/core/status_test.cc",
        "lib/core/stringpiece_test.cc",
        "lib/core/threadpool_test.cc",
        "lib/gtl/array_slice_test.cc",
        "lib/gtl/cleanup_test.cc",
        "lib/gtl/edit_distance_test.cc",
        "lib/gtl/flatmap_test.cc",
        "lib/gtl/flatset_test.cc",
        "lib/gtl/inlined_vector_test.cc",
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
        "lib/monitoring/metric_def_test.cc",
        "lib/monitoring/sampler_test.cc",
        "lib/random/distribution_sampler_test.cc",
        "lib/random/philox_random_test.cc",
        "lib/random/random_distributions_test.cc",
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
        "platform/net_test.cc",
        "platform/port_test.cc",
        "platform/profile_utils/cpu_utils_test.cc",
        "platform/subprocess_test.cc",
    ],
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

cc_test(
    name = "lib_jpeg_jpeg_mem_unittest",
    srcs = ["lib/jpeg/jpeg_mem_unittest.cc"],
    data = glob(["lib/jpeg/testdata/*.jpg"]),
    deps = [
        ":jpeg_internal",
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

cc_test(
    name = "lib_random_weighted_picker_test",
    size = "medium",
    srcs = ["lib/random/weighted_picker_test.cc"],
    deps = [
        ":lib",
        ":lib_internal",
        ":lib_test_internal",
        ":test",
        ":test_main",
    ],
)

tf_cc_tests(
    name = "higher_level_tests",
    size = "small",
    srcs = [
        "common_runtime/device_set_test.cc",
        "common_runtime/optimization_registry_test.cc",
        "common_runtime/pending_counts_test.cc",
        "common_runtime/session_test.cc",
        "common_runtime/simple_placer_test.cc",
        "example/feature_util_test.cc",
        "framework/allocator_test.cc",
        "framework/attr_value_util_test.cc",
        "framework/bfloat16_test.cc",
        "framework/cancellation_test.cc",
        "framework/common_shape_fns_test.cc",
        "framework/function_test.cc",
        "framework/graph_def_util_test.cc",
        "framework/kernel_def_builder_test.cc",
        "framework/memory_types_test.cc",
        "framework/node_def_builder_test.cc",
        "framework/node_def_util_test.cc",
        "framework/op_compatibility_test.cc",
        "framework/op_def_builder_test.cc",
        "framework/op_def_util_test.cc",
        "framework/op_kernel_test.cc",
        "framework/op_registration_test.cc",
        "framework/partial_tensor_shape_test.cc",
        # "framework/rendezvous_test.cc",  # flaky b/30476344
        "framework/resource_mgr_test.cc",
        "framework/resource_op_kernel_test.cc",
        "framework/shape_inference_test.cc",
        "framework/shape_inference_testutil_test.cc",
        "framework/tensor_shape_test.cc",
        "framework/tensor_slice_test.cc",
        "framework/tensor_test.cc",
        "framework/tensor_util_test.cc",
        "framework/tracking_allocator_test.cc",
        "framework/types_test.cc",
        "framework/unique_tensor_references_test.cc",
        "graph/algorithm_test.cc",
        "graph/edgeset_test.cc",
        "graph/equal_graph_def_test.cc",
        "graph/graph_constructor_test.cc",
        "graph/graph_def_builder_test.cc",
        "graph/graph_partition_test.cc",
        "graph/graph_test.cc",
        "graph/node_builder_test.cc",
        "graph/optimizer_cse_test.cc",
        "graph/quantize_training_test.cc",
        "graph/subgraph_test.cc",
        "graph/tensor_id_test.cc",
        "graph/validate_test.cc",
        "util/bcast_test.cc",
        "util/command_line_flags_test.cc",
        "util/device_name_utils_test.cc",
        "util/events_writer_test.cc",
        "util/example_proto_fast_parsing_test.cc",
        "util/example_proto_helper_test.cc",
        "util/memmapped_file_system_test.cc",
        "util/presized_cuckoo_map_test.cc",
        "util/reporter_test.cc",
        "util/saved_tensor_slice_util_test.cc",
        "util/sparse/sparse_tensor_test.cc",
        "util/semver_test.cc",
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
        ":test",
        ":test_main",
        ":testlib",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:scope",
        "//tensorflow/cc:sendrecv_ops",
        "//tensorflow/core/kernels:ops_util",
        "//third_party/eigen3",
    ],
)

tf_cc_tests_gpu(
    name = "gpu_related_tests",
    size = "small",
    srcs = glob(["user_ops/**/*_test.cc"]) + [
        "common_runtime/gpu/gpu_bfc_allocator_test.cc",
        "common_runtime/gpu/gpu_event_mgr_test.cc",
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
    name = "cuda_libdevice_path_test",
    size = "small",
    srcs = ["platform/cuda_libdevice_path_test.cc"],
    linkstatic = tf_kernel_tests_linkstatic(),
    tags = tf_cuda_tests_tags(),
    deps = [
        ":cuda_libdevice_path",
        ":lib",
        ":test",
        ":test_main",
    ],
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
        "//tensorflow/core/kernels:bcast_ops",
        "//tensorflow/core/kernels:cast_op",
        "//tensorflow/core/kernels:concat_op",
        "//tensorflow/core/kernels:identity_op",
        "//tensorflow/core/kernels:immutable_constant_op",
        "//tensorflow/core/kernels:matmul_op",
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
        "//tensorflow/cc:scope",
        "//tensorflow/core/kernels:array",
        "//tensorflow/core/kernels:math",
        "//third_party/eigen3",
    ],
)

tf_cc_test(
    name = "common_runtime_direct_session_test",
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
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core/kernels:control_flow_ops",
        "//tensorflow/core/kernels:cwise_op",
        "//tensorflow/core/kernels:dense_update_ops",
        "//tensorflow/core/kernels:fifo_queue_op",
        "//tensorflow/core/kernels:function_ops",
        "//tensorflow/core/kernels:identity_op",
        "//tensorflow/core/kernels:matmul_op",
        "//tensorflow/core/kernels:ops_util",
        "//tensorflow/core/kernels:queue_ops",
        "//tensorflow/core/kernels:session_ops",
        "//tensorflow/core/kernels:variable_ops",
        "//third_party/eigen3",
    ],
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
        "//tensorflow/core/kernels:control_flow_ops",
        "//tensorflow/core/kernels:cwise_op",
        "//tensorflow/core/kernels:dense_update_ops",
        "//tensorflow/core/kernels:fifo_queue_op",
        "//tensorflow/core/kernels:function_ops",
        "//tensorflow/core/kernels:identity_op",
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
    name = "common_runtime_function_test",
    size = "small",
    srcs = ["common_runtime/function_test.cc"],
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
        "//tensorflow/core/kernels:cast_op",
        "//tensorflow/core/kernels:cwise_op",
        "//tensorflow/core/kernels:function_ops",
        "//tensorflow/core/kernels:matmul_op",
        "//tensorflow/core/kernels:shape_ops",
        "//third_party/eigen3",
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
        "ops/sparse_ops_test.cc",
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

cc_test(
    name = "example_example_parser_configuration_test",
    size = "small",
    srcs = ["example/example_parser_configuration_test.cc"],
    data = [":example_parser_configuration_testdata"],
    deps = [
        ":core",
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
        "//tensorflow/core/kernels:ops_util",
    ],
)

tf_cc_test_gpu(
    name = "gpu_tracer_test",
    size = "small",
    srcs = ["common_runtime/gpu/gpu_tracer_test.cc"],
    args = ["--heap_check=local"],
    linkstatic = tf_kernel_tests_linkstatic(),
    tags = tf_cuda_tests_tags() + ["nomac"],
    deps = [
        ":all_kernels",
        ":core_cpu",
        ":core_cpu_internal",
        ":direct_session",
        ":direct_session_internal",
        ":framework",
        ":framework_internal",
        ":gpu_runtime",
        ":gpu_tracer",
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
        # GIF data
        "lib/gif/testdata/scan.gif",
        # GIF data with optimization
        "lib/gif/testdata/optimized.gif",
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
    srcs = ["platform/cuda_libdevice_path.cc"] + tf_additional_libdevice_srcs(),
    hdrs = ["platform/cuda_libdevice_path.h"],
    copts = tf_copts(),
    data = tf_additional_libdevice_data(),
    visibility = ["//visibility:public"],
    deps = [
        ":lib",
    ] + tf_additional_libdevice_deps(),
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

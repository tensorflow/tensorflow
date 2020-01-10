# Description:
# TensorFlow is a computational framework, primarily for use in machine
# learning applications.

package(default_visibility = ["//tensorflow:internal"])

package_group(name = "friends")

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("/tensorflow/tensorflow", "tf_copts")
load("/tensorflow/tensorflow", "tf_cc_tests")
load("/tensorflow/tensorflow", "tf_cuda_library")
load("/tensorflow/tensorflow", "tf_gen_op_libs")
load("/tensorflow/tensorflow", "tf_gpu_kernel_library")

# For platform specific build config
load(
    "/tensorflow/core/platform/default/build_config",
    "tf_proto_library",
    "tf_additional_lib_srcs",
    "tf_additional_test_srcs",
    "tf_kernel_tests_linkstatic",
)
load(
    "/tensorflow/core/platform/default/build_config_root",
    "tf_cuda_tests_tags",
)

cc_library(
    name = "lib",
    srcs = glob(
        [
            "lib/**/*.h",
            "lib/**/*.cc",
            "platform/*.h",
            "platform/*.cc",
            "public/*.h",
        ] + tf_additional_lib_srcs(),
        exclude = [
            "**/*test*",
        ],
    ),
    copts = tf_copts(),
    visibility = [
        ":friends",
        "//tensorflow:internal",
    ],
    deps = [
        ":protos_cc",
        "//tensorflow/core/platform/default/build_config:platformlib",
    ],
)

tf_cuda_library(
    name = "core_cpu",
    srcs = glob(
        [
            "common_runtime/**/*.h",
            "client/**/*.cc",
            "common_runtime/**/*.cc",
            "graph/**/*.h",
            "graph/**/*.cc",
        ],
        exclude = [
            "**/*test*",
            "**/*main.cc",
            "common_runtime/gpu/*.cc",
            "common_runtime/copy_tensor.cc",
            "common_runtime/gpu_device_factory.cc",
            "common_runtime/local_session.cc",
            "common_runtime/local_session.h",
        ],
    ),
    hdrs = glob(["public/**/*.h"]),
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":copy_tensor",
        ":framework",
        ":lib",
        ":protos_cc",
        "//third_party/eigen3",
    ],
    alwayslink = 1,
)

tf_cuda_library(
    name = "framework",
    srcs = glob(
        [
            "framework/**/*.h",
            "framework/**/*.cc",
            "util/**/*.h",
            "util/**/*.cc",
        ],
        exclude = [
            "**/*test*",
            "**/*main.cc",
        ],
    ),
    hdrs = glob(["public/**/*.h"]),
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":lib",
        ":protos_cc",
        "//third_party/eigen3",
    ],
    alwayslink = 1,
)

tf_cuda_library(
    name = "local",
    srcs = [
        "common_runtime/local_session.cc",
        "common_runtime/local_session.h",
    ],
    copts = tf_copts(),
    cuda_deps = [
        ":cuda",
    ],
    linkstatic = 1,
    deps = [
        ":core",
        ":lib",
    ],
    alwayslink = 1,
)

cc_library(
    name = "copy_tensor",
    deps = [
        ":lib",
        ":protos_cc",
        ":stream_executor",
        "//third_party/eigen3",
    ],
)

tf_cuda_library(
    name = "gpu_runtime",
    srcs = glob(
        [
            "common_runtime/gpu/**/*.h",
            "common_runtime/gpu/**/*.cc",
        ],
        exclude = [
            "**/*main.cc",
            "**/*test.cc",
        ],
    ),
    copts = tf_copts(),
    cuda_deps = [
        ":cuda",
    ],
    linkstatic = 1,
    deps = [
        ":core_cpu",
        ":lib",
        ":protos_cc",
        ":stream_executor",
        "//third_party/eigen3",
    ],
    alwayslink = 1,
)

# Test support library needed for higher-level tests
cc_library(
    name = "testlib",
    testonly = 1,
    srcs = [
        "common_runtime/kernel_benchmark_testlib.cc",
        "common_runtime/kernel_benchmark_testlib.h",
        "framework/function_testlib.cc",
        "framework/function_testlib.h",
        "framework/tensor_testutil.cc",
        "framework/tensor_testutil.h",
        "graph/testlib.cc",
        "graph/testlib.h",
    ],
    copts = tf_copts(),
    visibility = [
        ":friends",
        "//tensorflow:internal",
    ],
    deps = [
        ":core_cpu",
        ":tensorflow",
        ":test",
        "//tensorflow/core/platform/default/build_config:gtest",
    ],
)

tf_cuda_library(
    name = "tensorflow_opensource",
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":core",
        ":gpu_runtime",
        ":kernels",
        ":lib",
        ":local",
    ],
)

tf_cuda_library(
    name = "kernels",
    srcs = glob(
        [
            "kernels/**/*.h",
            "kernels/**/*.cc",
            "ops/**/*.h",
            "ops/**/*.cc",
            "user_ops/**/*.h",
            "user_ops/**/*.cc",
        ],
        exclude = [
            "**/*test*",
            "**/*main.cc",
            "kernels/**/*.cu.cc",
            "user_ops/**/*.cu.cc",
        ],
    ),
    copts = tf_copts(),
    cuda_deps = [
        ":gpu_kernels",
        ":cuda",
    ],
    linkstatic = 0,
    visibility = ["//visibility:public"],
    deps = [
        "@gemmlowp//:eight_bit_int_gemm",
        ":core",
        ":lib",
        ":protos_cc",
        ":stream_executor",
        "//tensorflow/models/embedding:word2vec_kernels",
        "//tensorflow/models/embedding:word2vec_ops",
        "//third_party/eigen3",
    ],
    alwayslink = 1,
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
        "//third_party/eigen3",
    ],
)

# Test support library needed for all tests
cc_library(
    name = "test",
    testonly = 1,
    srcs = [
        "platform/test.cc",
    ] + tf_additional_test_srcs(),
    hdrs = [
        "platform/test.h",
        "platform/test_benchmark.h",
    ],
    copts = tf_copts(),
    linkopts = ["-lm"],
    deps = [
        ":lib",
        "//tensorflow/core/platform/default/build_config:gtest",
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
        ":test",
        "//tensorflow/core/platform/default/build_config:test_main",
    ],
)

# TODO(opensource): Make it work externally
tf_proto_library(
    name = "protos_all",
    srcs = glob(["**/*.proto"]),
    cc_api_version = 2,
    go_api_version = 2,
    java_api_version = 2,
    py_api_version = 2,
    visibility = ["//tensorflow:internal"],
)

cc_library(
    name = "protos_cc",
    deps = ["//tensorflow/core/platform/default/build_config:protos_cc"],
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
        "math_ops",
        "nn_ops",
        "no_op",
        "parsing_ops",
        "random_ops",
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
        ":test_main",
    ],
)

cc_test(
    name = "lib_jpeg_jpeg_mem_unittest",
    srcs = ["lib/jpeg/jpeg_mem_unittest.cc"],
    data = glob(["lib/jpeg/testdata/*.jpg"]),
    deps = [
        ":lib",
        ":test_main",
    ],
)

cc_test(
    name = "lib_strings_ordered_code_test",
    srcs = ["lib/strings/ordered_code_test.cc"],
    copts = ["$(STACK_FRAME_UNLIMITED)"],  # Tests initialize large vectors
    deps = [
        ":lib",
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
        ],
        exclude = [
            # TODO(opensource): fix
            "common_runtime/gpu/*_test.cc",
            # Run by tests below
            "common_runtime/gpu/gpu_region_allocator_test.cc",
            "common_runtime/gpu/gpu_bfc_allocator_test.cc",
        ],
    ),
    deps = [
        ":core",
        ":kernels",
        ":lib",
        ":local",
        ":test_main",
        ":testlib",
        "//tensorflow/cc:cc_ops",
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
    ),
    deps = [
        ":kernels",
        ":local",
        ":test_main",
        ":testlib",
        "//tensorflow/cc:cc_ops",
    ],
)

tf_cuda_library(
    name = "stream_executor",
    deps = [
        "//tensorflow/core/platform/default/build_config:stream_executor",
    ],
)

cc_library(
    name = "cuda",
    visibility = [
        ":friends",
        "//tensorflow:internal",
    ],
    deps = [
        "//tensorflow/core/platform/default/build_config:cuda",
    ],
)

cc_library(
    name = "tensorflow",
    visibility = ["//visibility:public"],
    deps = [
        "tensorflow_opensource",
        "//tensorflow/core/platform/default/build_config:tensorflow_platform_specific",
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

# Android-specific BUILD targets
load("/tensorflow/tensorflow", "tf_android_core_proto_sources")

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
            "kernels/reference_gemm.h",
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
            "platform/google/stream_executor_util.h",
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
        "//tensorflow/core:kernels/fill_functor.h",
        "//tensorflow/core:kernels/gather_op.cc",
        "//tensorflow/core:kernels/identity_op.cc",
        "//tensorflow/core:kernels/identity_op.h",
        "//tensorflow/core:kernels/matmul_op.cc",
        "//tensorflow/core:kernels/matmul_op.h",
        "//tensorflow/core:kernels/no_op.cc",
        "//tensorflow/core:kernels/no_op.h",
        "//tensorflow/core:kernels/pack_op.cc",
        "//tensorflow/core:kernels/reference_gemm.h",
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
filegroup(
    name = "android_extended_ops",
    srcs = [
        "//tensorflow/core:kernels/avgpooling_op.cc",
        "//tensorflow/core:kernels/avgpooling_op.h",
        "//tensorflow/core:kernels/control_flow_ops.cc",
        "//tensorflow/core:kernels/control_flow_ops.h",
        "//tensorflow/core:kernels/conv_2d.h",
        "//tensorflow/core:kernels/conv_ops.cc",
        "//tensorflow/core:kernels/cwise_op_add.cc",
        "//tensorflow/core:kernels/cwise_op_div.cc",
        "//tensorflow/core:kernels/cwise_op_exp.cc",
        "//tensorflow/core:kernels/cwise_op_log.cc",
        "//tensorflow/core:kernels/cwise_op_mul.cc",
        "//tensorflow/core:kernels/cwise_op_sigmoid.cc",
        "//tensorflow/core:kernels/cwise_op_sqrt.cc",
        "//tensorflow/core:kernels/cwise_op_square.cc",
        "//tensorflow/core:kernels/cwise_op_sub.cc",
        "//tensorflow/core:kernels/cwise_op_tanh.cc",
        "//tensorflow/core:kernels/lrn_op.cc",
        "//tensorflow/core:kernels/maxpooling_op.cc",
        "//tensorflow/core:kernels/maxpooling_op.h",
        "//tensorflow/core:kernels/reduction_ops.h",
        "//tensorflow/core:kernels/reduction_ops_common.h",
        "//tensorflow/core:kernels/reduction_ops_max.cc",
        "//tensorflow/core:kernels/reduction_ops_min.cc",
        "//tensorflow/core:kernels/reduction_ops_sum.cc",
        "//tensorflow/core:kernels/relu_op.cc",
        "//tensorflow/core:kernels/relu_op.h",
        "//tensorflow/core:kernels/softplus_op.cc",
        "//tensorflow/core:kernels/softplus_op.h",
        "//tensorflow/core:kernels/transpose_op.cc",
        "//tensorflow/core:kernels/transpose_op.h",
        "//tensorflow/core:kernels/transpose_op_functor.h",
    ],
    visibility = ["//visibility:public"],
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

# For portable_proto_library

# Native library support for Android applications.
# Should be built to target Android with flag --copt=-mfpu=neon.
cc_library(
    name = "android_tensorflow_lib",
    srcs = [
        "//tensorflow/core:android_core_ops",
        "//tensorflow/core:android_extended_ops",
        "//tensorflow/core:android_srcs",
    ],
    copts = [
        "-mfpu=neon",
        "-std=c++11",
    ],
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

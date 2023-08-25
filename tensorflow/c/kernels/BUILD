load("//tensorflow:tensorflow.default.bzl", "filegroup", "tf_gen_op_libs", "tf_kernel_library")
load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load(
    "//tensorflow/core/tpu:build_defs.bzl",
    "if_libtpu_tf_status",
    "if_libtpu_tf_tensor",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

tf_kernel_library(
    name = "bitcast_op",
    prefix = "bitcast_op",
    deps = [
        "//tensorflow/c:c_api_headers",
        "//tensorflow/c:kernels",
        "//tensorflow/c:ops",
        "//tensorflow/c:tf_datatype",
        "//tensorflow/c:tf_status",
        "//tensorflow/c:tf_tensor",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
)

tf_kernel_library(
    name = "summary_op",
    prefix = "summary_op",
    deps = [
        ":tensor_shape_utils",
        "//tensorflow/c:c_api_headers",
        "//tensorflow/c:kernels",
        "//tensorflow/c:tf_status",
        "//tensorflow/c:tf_tensor",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//third_party/eigen3",
    ],
)

tf_kernel_library(
    name = "histogram_summary_op",
    prefix = "histogram_summary_op",
    deps = [
        "//tensorflow/c:c_api_headers",
        "//tensorflow/c:kernels",
        "//tensorflow/c:tf_status",
        "//tensorflow/c:tf_tensor",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//third_party/eigen3",
    ],
)

tf_kernel_library(
    name = "merge_summary_op",
    prefix = "merge_summary_op",
    deps = [
        "//tensorflow/c:c_api_headers",
        "//tensorflow/c:kernels",
        "//tensorflow/c:tf_status",
        "//tensorflow/c:tf_tensor",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ],
)

tf_gen_op_libs(
    op_lib_names = ["bitcast"],
    deps = [
        "//tensorflow/c:c_api_headers",
        "//tensorflow/c:ops",
        "//tensorflow/c:tf_datatype",
        "//tensorflow/core:lib",
    ] + if_libtpu_tf_status() + if_libtpu_tf_tensor(),
)

tf_gen_op_libs(
    op_lib_names = ["summary"],
    deps = [
        "//tensorflow/c:c_api_headers",
        "//tensorflow/c:ops",
        "//tensorflow/c:tf_status",
        "//tensorflow/core:lib",
    ],
)

tf_gen_op_libs(
    op_lib_names = ["histogram_summary"],
    deps = [
        "//tensorflow/c:c_api_headers",
        "//tensorflow/c:ops",
        "//tensorflow/c:tf_status",
        "//tensorflow/core:lib",
    ],
)

tf_gen_op_libs(
    op_lib_names = ["merge_summary"],
    deps = [
        "//tensorflow/c:c_api_headers",
        "//tensorflow/c:ops",
        "//tensorflow/c:tf_status",
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "bitcast_op_test",
    srcs = ["bitcast_op_test.cc"],
    deps = [
        ":bitcast_op",
        "//tensorflow/c:c_api",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test(
    name = "summary_op_test",
    srcs = ["summary_op_test.cc"],
    deps = [
        ":summary_op",
        "//tensorflow/c:c_api",
        "//tensorflow/c:kernels",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test(
    name = "summary_op_benchmark_test",
    size = "small",
    srcs = ["summary_op_benchmark_test.cc"],
    deps = [
        ":summary_op",
        "//tensorflow/c:c_api",
        "//tensorflow/c:kernels",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

cc_library(
    name = "tensor_shape_utils",
    srcs = ["tensor_shape_utils.cc"],
    hdrs = ["tensor_shape_utils.h"],
    visibility = ["//visibility:private"],
    deps = [
        "//tensorflow/c:tf_tensor",
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "tensor_shape_utils_test",
    srcs = ["tensor_shape_utils_test.cc"],
    deps = [
        ":tensor_shape_utils",
        "//tensorflow/c:tf_tensor_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

# Be aware that this target is included by
# //third_party/tensorflow/core/kernels:portable_core_ops
#
filegroup(
    name = "portable_all_op_kernels",
    srcs = [
        "bitcast_op.cc",
        "histogram_summary_op.cc",
        "merge_summary_op.cc",
        "summary_op.cc",
        "tensor_shape_utils.cc",
        "tensor_shape_utils.h",
    ],
)

filegroup(
    name = "portable_all_ops",
    srcs = [
        "ops/bitcast.cc",
        "ops/histogram_summary.cc",
        "ops/merge_summary.cc",
        "ops/summary.cc",
    ],
)

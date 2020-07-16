load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
    "tf_gen_op_libs",
    "tf_kernel_library",
)

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

tf_kernel_library(
    name = "bitcast_op",
    prefix = "bitcast_op",
    deps = [
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
        "//tensorflow/c/kernels:tensor_shape_utils",
        "//tensorflow/c:kernels",
        "//tensorflow/c:tf_tensor",
        "//tensorflow/core:framework"
    ],
)


tf_gen_op_libs(
    op_lib_names = ["bitcast"],
    deps = [
        "//tensorflow/c:ops",
        "//tensorflow/c:tf_datatype",
        "//tensorflow/c:tf_status",
        "//tensorflow/c:tf_tensor",
        "//tensorflow/core:lib",
    ],
)

tf_gen_op_libs(
    op_lib_names = ["summary"],
    deps = [
        "//tensorflow/c:ops",
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "bitcast_op_test",
    srcs = ["bitcast_op_test.cc"],
    deps = [
        ":bitcast_op",
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
        ":summary_op_lib",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib"
    ],
)

cc_library(
    name = "tensor_shape_utils",
    srcs = ["tensor_shape_utils.cc",],
    hdrs = ["tensor_shape_utils.h",],
    deps = [ "//tensorflow/c:tf_tensor",], 
    visibility = ["//visibility:private"],
)

tf_cc_test(
    name = "tensor_shape_utils_test",
    srcs = ["tensor_shape_utils_test.cc"],
    deps = [
        ":tensor_shape_utils",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

# Changes to the Android srcs here should be replicated in
# tensorflow/contrib/makefile/tf_op_files.txt.
#
# Be aware that this target is included by
# //third_party/tensorflow/core/kernels:android_core_ops
#
# LINT.IfChange
filegroup(
    name = "android_all_op_kernels",
    srcs = [
        "bitcast_op.cc", 
        "summary_op.cc"
    ],
)
# LINT.ThenChange(//tensorflow/contrib/makefile/tf_op_files.txt)

filegroup(
    name = "android_all_ops",
    srcs = [
        "ops/bitcast.cc", 
        "ops/summary.cc"
    ],
)

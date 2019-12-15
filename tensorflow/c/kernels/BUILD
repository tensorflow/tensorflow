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
    ],
)
# LINT.ThenChange(//tensorflow/contrib/makefile/tf_op_files.txt)

filegroup(
    name = "android_all_ops",
    srcs = ["ops/bitcast.cc"],
)

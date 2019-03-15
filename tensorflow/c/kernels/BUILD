load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
    "tf_kernel_library",
)

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

tf_kernel_library(
    name = "bitcast_op",
    prefix = "bitcast_op",
    deps = [
        "//tensorflow/c:kernels",
        "//tensorflow/core:framework",
        "//tensorflow/core:ops",
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
    ],
)

# Changes to the Android srcs here should be replicated in
# tensorflow/contrib/makefile/tf_op_files.txt
# LINT.IfChange
filegroup(
    name = "android_all_ops",
    srcs = [
        "bitcast_op.cc",
    ],
)
# LINT.ThenChange(//tensorflow/contrib/makefile/tf_op_files.txt)

load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
    "tf_cuda_library",
)

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

tf_cuda_library(
    name = "bitcast_op",
    srcs = ["bitcast_op.cc"],
    deps = [
        "//tensorflow/c:kernels",
        "//tensorflow/core:framework",
        "//tensorflow/core:ops",
        "@com_google_absl//absl/strings:str_format",
    ],
    alwayslink = 1,
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

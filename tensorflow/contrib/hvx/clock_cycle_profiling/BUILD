# Description:
#   contains parts of TensorFlow that are experimental or unstable and which are not supported.

load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_binary",
    "tf_copts",
)

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["LICENSE"])

tf_cc_binary(
    name = "clock_cycle_profiling",
    testonly = 1,
    srcs = ["clock_cycle_profiling_main.cc"],
    copts = tf_copts(),
    linkopts = select({
        "//tensorflow:android": [
            "-pie",
            "-s",
            "-landroid",
            "-ljnigraphics",
            "-llog",
            "-lm",
            "-z defs",
            "-s",
            "-Wl,--exclude-libs,ALL",  # Exclude syms in all libs from auto export
        ],
        "//conditions:default": [],
    }),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = select({
        "//tensorflow:android": [
            "//tensorflow/core:android_tensorflow_lib",
            "//tensorflow/core:android_tensorflow_test_lib",
        ],
        "//conditions:default": [
            "//tensorflow/core:lib",
            "//tensorflow/core:framework_internal",
            "//tensorflow/core:tensorflow",
        ],
    }),
)

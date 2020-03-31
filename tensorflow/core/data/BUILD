load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("//tensorflow/core/platform:build_config.bzl", "tf_protos_all")

package(
    default_visibility = [
        "//tensorflow:internal",
    ],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["LICENSE"])

cc_library(
    name = "standalone",
    srcs = ["standalone.cc"],
    hdrs = ["standalone.h"],
    deps = [
        "//tensorflow/core:all_kernels",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:session_options",
        "@com_google_absl//absl/memory",
    ],
)

tf_cc_test(
    name = "standalone_test",
    srcs = ["standalone_test.cc"],
    deps = [
        ":standalone",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ] + tf_protos_all(),
)

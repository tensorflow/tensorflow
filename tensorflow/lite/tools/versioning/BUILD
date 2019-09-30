load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
)

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "op_version",
    srcs = ["op_version.cc"],
    hdrs = [
        "op_version.h",
    ],
    deps = [
        "//tensorflow/core:tflite_portable_logging",
        "//tensorflow/lite/kernels/internal:compatibility",
        "//tensorflow/lite/schema:schema_fbs_with_mutable",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@flatbuffers",
    ],
)

tf_cc_test(
    name = "op_version_test",
    srcs = ["op_version_test.cc"],
    deps = [
        ":op_version",
        "//tensorflow/lite/schema:schema_fbs_with_mutable",
        "@com_google_googletest//:gtest_main",
    ],
)

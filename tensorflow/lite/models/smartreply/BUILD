package(default_visibility = [
    "//visibility:public",
])

load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("//tensorflow/lite:build_def.bzl", "gen_selected_ops", "tflite_copts")

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

gen_selected_ops(
    name = "smartreply_ops",
    model = "@tflite_smartreply//:smartreply.tflite",
)

cc_library(
    name = "custom_ops",
    srcs = [
        "ops/extract_feature.cc",
        "ops/normalize.cc",
        "ops/predict.cc",
        ":smartreply_ops",
    ],
    copts = tflite_copts(),
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite:string_util",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/kernels:kernel_util",
        "@com_google_absl//absl/strings",
        "@com_googlesource_code_re2//:re2",
        "@farmhash_archive//:farmhash",
    ],
    alwayslink = 1,
)

cc_library(
    name = "predictor_lib",
    srcs = ["predictor.cc"],
    hdrs = ["predictor.h"],
    copts = tflite_copts(),
    deps = [
        ":custom_ops",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:string_util",
        "//tensorflow/lite/kernels:builtin_ops",
        "@com_google_absl//absl/strings",
        "@com_googlesource_code_re2//:re2",
    ],
)

tf_cc_test(
    name = "predictor_test",
    srcs = ["predictor_test.cc"],
    data = [
        "//tensorflow/lite/models:testdata/smartreply_samples.tsv",
        "@tflite_smartreply//:smartreply.tflite",
    ],
    tags = ["no_oss"],
    deps = [
        ":predictor_lib",
        "//tensorflow/core:test",
        "//tensorflow/lite:string_util",
        "//tensorflow/lite/testing:util",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "extract_feature_op_test",
    size = "small",
    srcs = ["ops/extract_feature_test.cc"],
    tags = ["no_oss"],
    deps = [
        ":custom_ops",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
        "@farmhash_archive//:farmhash",
    ],
)

tf_cc_test(
    name = "normalize_op_test",
    size = "small",
    srcs = ["ops/normalize_test.cc"],
    tags = ["no_oss"],
    deps = [
        ":custom_ops",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:string_util",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "predict_op_test",
    size = "small",
    srcs = ["ops/predict_test.cc"],
    tags = ["no_oss"],
    deps = [
        ":custom_ops",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:string_util",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)

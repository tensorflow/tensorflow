package(default_visibility = ["//visibility:public"])

load("//tensorflow/lite:build_def.bzl", "tflite_copts", "gen_selected_ops")

licenses(["notice"])  # Apache 2.0

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
        "@com_google_absl//absl/strings",
        "@com_googlesource_code_re2//:re2",
        "@farmhash_archive//:farmhash",
    ],
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

cc_test(
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

cc_test(
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

cc_test(
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

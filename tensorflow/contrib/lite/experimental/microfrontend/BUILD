package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "tf_cc_test")

cc_library(
    name = "audio_microfrontend",
    srcs = ["audio_microfrontend.cc"],
    deps = [
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite/experimental/microfrontend/lib:frontend",
        "//tensorflow/contrib/lite/kernels:kernel_util",
        "//tensorflow/contrib/lite/kernels/internal:reference",
        "@flatbuffers",
    ],
)

tf_cc_test(
    name = "audio_microfrontend_test",
    size = "small",
    srcs = ["audio_microfrontend_test.cc"],
    tags = ["tflite_not_portable_ios"],
    deps = [
        ":audio_microfrontend",
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

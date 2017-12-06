package(default_visibility = [
    "//visibility:public",
])

licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "tf_cc_binary")
load("//tensorflow/contrib/lite:build_def.bzl", "tflite_linkopts")

exports_files(glob([
    "testdata/*.bmp",
]))

tf_cc_binary(
    name = "label_image",
    srcs = ["label_image.cc"],
    linkopts = tflite_linkopts() + ["-lm"],
    deps = [
        ":bitmap_helpers",
        ":get_top_n",
        "//tensorflow/contrib/lite/kernels:builtin_ops",
    ],
)

cc_library(
    name = "bitmap_helpers",
    srcs = ["bitmap_helpers.cc"],
    hdrs = ["label_image.h"],
)

cc_library(
    name = "get_top_n",
    srcs = ["get_top_n.cc"],
    hdrs = ["label_image.h"],
)

cc_test(
    name = "label_image_test",
    srcs = ["label_image_test.cc"],
    data = [
        "testdata/lena.bmp",
    ],
    deps = [
        ":bitmap_helpers",
        ":get_top_n",
        "@com_google_googletest//:gtest",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)

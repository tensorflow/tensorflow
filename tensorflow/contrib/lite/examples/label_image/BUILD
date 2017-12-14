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
    srcs = [
        "get_top_n.h",
        "get_top_n_impl.h",
        "label_image.cc",
    ],
    linkopts = tflite_linkopts() + select({
        "//tensorflow:android": [
            "-pie",  # Android 5.0 and later supports only PIE
            "-lm",  # some builtin ops, e.g., tanh, need -lm
        ],
        "//conditions:default": [],
    }),
    deps = [
        ":bitmap_helpers",
        "//tensorflow/contrib/lite/kernels:builtin_ops",
    ],
)

cc_library(
    name = "bitmap_helpers",
    srcs = ["bitmap_helpers.cc"],
    hdrs = [
        "bitmap_helpers.h",
        "bitmap_helpers_impl.h",
        "label_image.h",
    ],
)

cc_test(
    name = "label_image_test",
    srcs = [
        "get_top_n.h",
        "get_top_n_impl.h",
        "label_image_test.cc",
    ],
    data = [
        "testdata/grace_hopper.bmp",
    ],
    deps = [
        ":bitmap_helpers",
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

package(default_visibility = [
    "//visibility:public",
])

licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "tf_cc_binary")

tf_cc_binary(
    name = "label_image",
    srcs = ["label_image.cc"],
    deps = [
        "//tensorflow/contrib/lite/kernels:builtin_ops",
    ],
    linkopts = select({
        "//tensorflow:android": [
            "-s",
            "-pie",
            "-landroid",
            "-llog",
            "-lm",
            "-z defs",
            "-Wl,--exclude-libs,ALL",  # Exclude syms in all libs from auto export
        ],
        "//conditions:default": [],
    }),
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

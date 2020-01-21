load(
    "//tensorflow/core/platform:rules_cc.bzl",
    "cc_library",
)

package(
    default_visibility = [
        # tensorflow/core:lib effectively exposes all targets under tensorflow/core/lib/**
        "//tensorflow/core:__pkg__",
    ],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "png_io",
    srcs = ["png_io.cc"],
    hdrs = ["png_io.h"],
    deps = [
        "//tensorflow/core/platform:byte_order",
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/platform:png",
        "//tensorflow/core/platform:stringpiece",
        "//tensorflow/core/platform:types",
        "@com_google_absl//absl/base",
        "@zlib_archive//:zlib",
    ],
)

filegroup(
    name = "testdata",
    srcs = [
        "testdata/lena_gray.png",
        "testdata/lena_palette.png",
        "testdata/lena_palette_trns.png",
        "testdata/lena_rgba.png",
    ],
)

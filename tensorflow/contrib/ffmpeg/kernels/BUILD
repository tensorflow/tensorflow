# Description:
#   Libraries and kernels for manipulating audio and video using FFmpeg.
#   (https://www.ffmpeg.org)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

cc_library(
    name = "ffmpeg_lib",
    srcs = ["ffmpeg_lib.cc"],
    hdrs = ["ffmpeg_lib.h"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
)

cc_test(
    name = "ffmpeg_lib_installed_test",
    srcs = ["ffmpeg_lib_test.cc"],
    args = [
        "--should_ffmpeg_be_installed=true",
    ],
    data = [
        ":testdata/test_sound1.mp3",
    ],
    tags = [
        "local",
        "manual",
    ],
    deps = [
        ":ffmpeg_lib",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
    ],
)

cc_test(
    name = "ffmpeg_lib_uninstalled_test",
    srcs = ["ffmpeg_lib_test.cc"],
    args = [
        "--should_ffmpeg_be_installed=false",
    ],
    data = [
        ":testdata/test_sound1.mp3",
    ],
    tags = [
        "local",
        "manual",
    ],
    deps = [
        ":ffmpeg_lib",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
    ],
)

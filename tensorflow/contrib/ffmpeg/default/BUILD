# Description:
#   Libraries and kernels for manipulating audio and video using FFmpeg.
#   (https://www.ffmpeg.org)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

cc_library(
    name = "ffmpeg_lib",
    srcs = ["ffmpeg_lib.cc"],
    hdrs = [
        # Header is shared between implementations.
        "//tensorflow/contrib/ffmpeg:ffmpeg_lib.h",
    ],
    deps = [
        "//tensorflow/core:framework_headers_lib",
        "@protobuf//:protobuf_headers",
    ],
)

cc_test(
    name = "ffmpeg_lib_installed_test",
    srcs = ["ffmpeg_lib_test.cc"],
    args = [
        "--should_ffmpeg_be_installed=true",
    ],
    data = [
        "//tensorflow/contrib/ffmpeg:test_data",
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

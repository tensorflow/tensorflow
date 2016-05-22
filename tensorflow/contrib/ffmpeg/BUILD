# Ops that process audio and/or video files using FFmpeg.
# (https://www.ffmpeg.org/)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

load("//tensorflow:tensorflow.bzl", "tf_gen_op_wrapper_py")
load("//tensorflow:tensorflow.bzl", "tf_copts")
load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")
load("//tensorflow:tensorflow.bzl", "tf_py_test")

filegroup(
    name = "test_data",
    srcs = glob(["testdata/*"]),
)

cc_library(
    name = "decode_audio_op_cc",
    srcs = ["decode_audio_op.cc"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:private"],
    deps = [
        "//tensorflow/contrib/ffmpeg/kernels:ffmpeg_lib",
        "//tensorflow/core:framework_headers_lib",
        "//third_party/eigen3",
    ],
    alwayslink = 1,
)

cc_library(
    name = "encode_audio_op_cc",
    srcs = ["encode_audio_op.cc"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:private"],
    deps = [
        "//tensorflow/contrib/ffmpeg/kernels:ffmpeg_lib",
        "//tensorflow/core:framework_headers_lib",
        "//third_party/eigen3",
    ],
    alwayslink = 1,
)

tf_custom_op_library(
    name = "ffmpeg.so",
    deps = [
        ":decode_audio_op_cc",
        ":encode_audio_op_cc",
    ],
)

tf_gen_op_wrapper_py(
    name = "decode_audio_op_py",
    require_shape_functions = True,
    visibility = ["//visibility:private"],
    deps = [
        ":decode_audio_op_cc",
    ],
)

tf_gen_op_wrapper_py(
    name = "encode_audio_op_py",
    require_shape_functions = True,
    visibility = ["//visibility:private"],
    deps = [
        ":encode_audio_op_cc",
    ],
)

tf_py_test(
    name = "decode_audio_op_test",
    srcs = ["decode_audio_op_test.py"],
    additional_deps = [
        ":ffmpeg_ops_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:platform",
    ],
    data = [
        ":ffmpeg.so",
        ":test_data",
    ],
    tags = [
        "local",
        "manual",
    ],
)

tf_py_test(
    name = "encode_audio_op_test",
    srcs = ["encode_audio_op_test.py"],
    additional_deps = [
        ":ffmpeg_ops_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:platform",
    ],
    data = [
        ":ffmpeg.so",
        ":test_data",
    ],
    tags = [
        "local",
        "manual",
    ],
)

py_library(
    name = "ffmpeg_ops_py",
    srcs = [
        "__init__.py",
        "ffmpeg_ops.py",
    ],
    data = [
        ":ffmpeg.so",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":decode_audio_op_py",
        ":encode_audio_op_py",
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

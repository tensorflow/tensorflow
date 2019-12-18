# Description:
# S3 support for TensorFlow.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_binary",
    "tf_cc_test",
)

tf_cc_binary(
    name = "libs3_file_system_shared.so",
    srcs = [
        "aws_crypto.cc",
        "aws_crypto.h",
        "aws_logging.cc",
        "aws_logging.h",
        "s3_file_system.cc",
        "s3_file_system.h",
    ],
    copts = ["-Wno-sign-compare"],
    defines = select({
        "//conditions:default": [
            "ENABLE_CURL_CLIENT",
            "ENABLE_NO_ENCRYPTION",
        ],
    }),
    linkshared = 1,
    deps = [
        "//tensorflow/core:framework_headers_lib",
        "@aws",
        "@com_google_protobuf//:protobuf_headers",
        "@curl",
    ],
)

cc_library(
    name = "aws_crypto",
    srcs = [
        "aws_crypto.cc",
    ],
    hdrs = [
        "aws_crypto.h",
    ],
    deps = [
        "@aws",
        "@boringssl//:crypto",
    ],
    alwayslink = 1,
)

cc_library(
    name = "aws_logging",
    srcs = [
        "aws_logging.cc",
    ],
    hdrs = [
        "aws_logging.h",
    ],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/platform:stringprintf",
        "@aws",
    ],
    alwayslink = 1,
)

cc_library(
    name = "s3_file_system",
    srcs = [
        "s3_file_system.cc",
    ],
    hdrs = [
        "s3_file_system.h",
    ],
    deps = [
        ":aws_crypto",
        ":aws_logging",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/platform:path",
        "//tensorflow/core/platform:str_util",
        "@aws",
    ],
    alwayslink = 1,
)

tf_cc_test(
    name = "s3_file_system_test",
    size = "small",
    srcs = [
        "s3_file_system_test.cc",
    ],
    tags = [
        "manual",
    ],
    deps = [
        ":s3_file_system",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform:path",
        "@aws",
    ],
)

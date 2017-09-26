# Description:
# S3 support for TensorFlow.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
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

cc_binary(
    name = "s3_file_system.so",
    srcs = [
        "s3_crypto.cc",
        "s3_crypto.h",
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
        "@aws//:aws",
        "@curl//:curl",
        "@protobuf_archive//:protobuf_headers",
    ],
)

cc_library(
    name = "s3_crypto",
    srcs = [
        "s3_crypto.cc",
    ],
    hdrs = [
        "s3_crypto.h",
    ],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "@aws//:aws",
        "@boringssl//:crypto",
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
        ":s3_crypto",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "@aws//:aws",
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
        "@aws//:aws",
    ],
)

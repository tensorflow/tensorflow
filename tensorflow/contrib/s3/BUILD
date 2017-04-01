# Description:
# S3 support for TensorFlow.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

cc_binary(
    name = "s3_file_system.so",
    srcs = [
        "s3_file_system.cc",
        "s3_crypto.cc",
        "s3_crypto.h",
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
        "@curl//:curl",
        "@protobuf//:protobuf_headers",
        "//tensorflow/core:framework_headers_lib",
        "//external:aws",
    ],
)

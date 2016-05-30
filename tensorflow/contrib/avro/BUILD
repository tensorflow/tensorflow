# Description:
#   Contains ops for reading and writing Apache Avro files.
#   (https://avro.apache.org/)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

load("//third_party/avro:build_defs.bzl", "avro_gen_cpp")

avro_gen_cpp(
    name = "example_h",
    srcs = ["example.json"],
    outs = ["example.h"],
    namespace = "tensorflow",
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

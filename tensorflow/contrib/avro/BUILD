package(default_visibility = ["//tensorflow:__subpackages__"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "tf_custom_op_py_library")
load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")
load("//tensorflow:tensorflow.bzl", "cc_header_only_library")
load("//tensorflow:tensorflow.bzl", "tf_gen_op_wrapper_py")

# TODO: Use tf_gen_op_wrapper_py to generate the code which should
# have already been generated in python/avro_record_dataset.py

tf_custom_op_py_library(
    name = "avro",
    srcs = glob(["python/*.py"]),
    dso = [":python/_avro_record_dataset.so",
           ":python/_parse_avro_record.so",
           ":python/libavro.so"
    ],
    srcs_version = "PY2",
    visibility = ["//visibility:public"],
)

# HACK: tensorflow will link to libavro.so correctly, but it will
# leave it an external bazel directory, which won't exist in the
# generated pip package. Therefore, we copy the libavro.so file to the
# same directory as these files.
genrule(
    name = "copy_libavro.so",
    srcs = ["@avro_c//:libavro_so"],
    outs = ["python/libavro.so"],
    cmd = "cp -f $< $@",
    output_to_bindir = 1,
)

tf_custom_op_library(
    name = "python/_avro_record_dataset.so",
    srcs = ["ops/avro_record_dataset.cc"],
    deps = ["@avro_c//:avro_c"]
)

tf_custom_op_library(
    name = "python/_parse_avro_record.so",
    srcs = ["ops/parse_avro_record.cc",
            "ops/avro_helper.cc",
            "ops/avro_helper.h"],
    deps = [
        "@avro_c//:avro_c",
    ]
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

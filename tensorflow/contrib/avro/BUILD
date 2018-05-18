package(default_visibility = ["//tensorflow:__subpackages__"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "tf_custom_op_py_library")
load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")
load("//tensorflow:tensorflow.bzl", "cc_header_only_library")
load("//tensorflow:tensorflow.bzl", "tf_gen_op_wrapper_py")
load("//tensorflow:tensorflow.bzl", "tf_gen_op_libs")

tf_custom_op_py_library(
    name = "avro",
    srcs = glob(["python/*.py"]),
    dso = [":python/_avro_record_dataset.so",
           ":python/_parse_avro_record.so",
           ":python/libavro.so"
    ],
    kernels = [
        ":avro_record_dataset_op_lib",
        ":parse_avro_record_op_lib",
    ],
    deps = [
         ":avro_record_dataset",
         ":parse_avro_record",
    ],
    srcs_version = "PY2",
    visibility = ["//visibility:public"],
)

# Tensorflow will link to libavro.so correctly, but it will leave it an external bazel
# directory, which won't exist in the generated pip package. Therefore, we copy libavro.so
# into the same directory -- of the pip package.
# TODO(fraudies): Remove this copy rule;
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
    srcs = ["ops/parse_avro_record.cc"],
    deps = [
        "@avro_c//:avro_c",
    ]
)

tf_gen_op_libs(
    op_lib_names = ["avro_record_dataset"],
    deps = [
         "@avro_c//:avro_c",
    ],
)

tf_gen_op_wrapper_py(
    name = "avro_record_dataset",
    deps = [
         ":avro_record_dataset_op_lib",
         "@avro_c//:avro_c",
    ],
)

tf_gen_op_libs(
    op_lib_names = ["parse_avro_record"],
    deps = [
         "@avro_c//:avro_c",
    ],
)

tf_gen_op_wrapper_py(
    name = "parse_avro_record",
    deps = [
         ":parse_avro_record_op_lib",
         "@avro_c//:avro_c",
    ],
)

py_test(
    name = "avro_record_dataset_test",
    size = "small",
    srcs = ["python/tests/avro_record_dataset_test.py"],
    deps = [
        ":avro",
        "//tensorflow/contrib/avro/python/utils:avro_utils",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:io_ops",
    ],
)

py_test(
    name = "parse_avro_record_test",
    size = "small",
    srcs = ["python/tests/parse_avro_record_test.py"],
    deps = [
        ":avro",
        "//tensorflow/contrib/avro/python/utils:avro_utils",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:io_ops",
        "//tensorflow/python:parsing_ops",
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

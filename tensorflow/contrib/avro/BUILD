package(default_visibility = ["//tensorflow:__subpackages__"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "tf_custom_op_py_library")
load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")
load("//tensorflow:tensorflow.bzl", "tf_gen_op_wrapper_py")
load("//tensorflow:tensorflow.bzl", "tf_gen_op_libs")

tf_custom_op_py_library(
    name = "avro",
    srcs = ["python/avro_record_dataset.py",
            "python/parse_avro_record.py",
            "//tensorflow/contrib/avro/python/utils:avro_utils"],
    dso = [":python/_avro_record_dataset.so",
           ":python/_parse_avro_record.so",
           "//third_party/avro:libavro_so"
    ],
    kernels = [
        ":avro_record_dataset_op_lib",
        ":parse_avro_record_op_lib",
    ],
    deps = [
         ":avro_record_dataset",
         ":parse_avro_record",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
)

tf_custom_op_library(
    name = "python/_avro_record_dataset.so",
    srcs = ["ops/avro_record_dataset.cc"],
    deps = ["//third_party/avro"],
)

tf_custom_op_library(
    name = "python/_parse_avro_record.so",
    srcs = ["ops/parse_avro_record.cc"],
    deps = [
        "//third_party/avro",
        "@com_googlesource_code_re2//:re2",
    ],
)

tf_gen_op_libs(
    op_lib_names = ["avro_record_dataset"],
    deps = [
         "//third_party/avro"
    ],
)

tf_gen_op_wrapper_py(
    name = "avro_record_dataset",
    deps = [
         ":avro_record_dataset_op_lib",
         "//third_party/avro"
    ],
)

tf_gen_op_libs(
    op_lib_names = ["parse_avro_record"],
    deps = [
         "//third_party/avro"
    ],
)

tf_gen_op_wrapper_py(
    name = "parse_avro_record",
    deps = [
         ":parse_avro_record_op_lib",
         "//third_party/avro"
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
        "//tensorflow/python/data/util:convert",
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

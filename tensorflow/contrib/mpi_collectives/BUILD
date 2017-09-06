# Ops that communicate with other processes via MPI.

package(default_visibility = [
    "//tensorflow:__subpackages__",
])

licenses(["notice"])  # Apache 2.0

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

load(
    "//tensorflow/core:platform/default/build_config.bzl",
    "tf_proto_library_cc",
)

tf_proto_library_cc(
    name = "mpi_message_proto",
    srcs = ["mpi_message.proto"],
    cc_api_version = 2,
    protodeps = ["//tensorflow/core:protos_all"],
    visibility = [
        "//tensorflow:__subpackages__",
    ],
)

load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")
load("//tensorflow:tensorflow.bzl", "tf_py_test")

tf_custom_op_library(
    name = "mpi_collectives.so",
    srcs = [
        "mpi_ops.cc",
        "ring.cc",
        "ring.h",
    ],
    gpu_srcs = [
        "ring.cu.cc",
        "ring.h",
    ],
    deps = [
        ":mpi_message_proto_cc",
        "//third_party/mpi",
    ],
)

tf_py_test(
    name = "mpi_ops_test",
    srcs = ["mpi_ops_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:platform",
    ],
    data = [
        ":mpi_collectives.so",
    ],
    tags = ["manual"],
)

py_library(
    name = "mpi_ops_py",
    srcs = [
        "__init__.py",
        "mpi_ops.py",
    ],
    data = [
        ":mpi_collectives.so",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
)

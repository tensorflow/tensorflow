# Ops that communicate with other processes via MPI.

package(default_visibility = [
    "//tensorflow:__subpackages__",
])

licenses(["notice"])  # Apache 2.0

load(
    "//tensorflow/core:platform/default/build_config.bzl",
    "tf_additional_mpi_lib_defines",
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

cc_library(
    name = "mpi_defines",
    defines = tf_additional_mpi_lib_defines(),
)

load(
    "//tensorflow:tensorflow.bzl",
    "tf_custom_op_py_library",
    "tf_custom_op_library",
    "tf_gen_op_wrapper_py",
    "tf_gen_op_libs",
    "tf_kernel_library",
    "tf_py_test",
)

tf_custom_op_library(
    name = "python/ops/_mpi_ops.so",
    srcs = [
        "kernels/mpi_ops.cc",
        "kernels/ring.cc",
        "kernels/ring.h",
        "ops/mpi_ops.cc",
    ],
    gpu_srcs = [
        "kernels/ring.cu.cc",
        "kernels/ring.h",
    ],
    deps = [
        ":mpi_defines",
        ":mpi_message_proto_cc",
        "//tensorflow/stream_executor:stream_executor_headers_lib",
        "//third_party/mpi",
    ],
)

tf_kernel_library(
    name = "mpi_ops_kernels",
    srcs = [
        "kernels/mpi_ops.cc",
        "kernels/ring.cc",
    ],
    hdrs = [
        "kernels/ring.h",
    ],
    gpu_srcs = [
        "kernels/ring.cu.cc",
    ],
    deps = [
        ":mpi_defines",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:gpu_headers_lib",
        "//tensorflow/core:lib",
        "//tensorflow/core:proto_text",
        "//tensorflow/core:stream_executor",
    ],
    # TODO: Include?    alwayslink = 1,
)

tf_gen_op_libs(
    op_lib_names = ["mpi_ops"],
)

tf_gen_op_wrapper_py(
    name = "mpi_ops",
    deps = [":mpi_ops_op_lib"],
)

tf_custom_op_py_library(
    name = "mpi_collectives_py",
    srcs = [
        "__init__.py",
        "python/ops/mpi_ops.py",
    ],
    dso = [
        ":python/ops/_mpi_ops.so",
    ],
    kernels = [
        ":mpi_ops_kernels",
        ":mpi_ops_op_lib",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":mpi_ops",
        "//tensorflow/contrib/util:util_py",
        "//tensorflow/python:device",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:util",
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
        ":python/ops/_mpi_ops.so",
    ],
    tags = ["manual"],
)

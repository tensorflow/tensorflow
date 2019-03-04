package(default_visibility = ["//tensorflow:internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load(
    "//tensorflow:tensorflow.bzl",
    "if_not_windows",
    "if_windows",
    "tf_custom_op_library",
    "tf_custom_op_py_library",
    "tf_gen_op_libs",
    "tf_gen_op_wrapper_py",
    "tf_kernel_library",
    "tf_py_test",
)

py_library(
    name = "ignite",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":dataset_ops",
        ":igfs_ops",
    ],
)

tf_custom_op_library(
    name = "_ignite_ops.so",
    srcs = [
        "kernels/igfs/igfs.h",
        "ops/dataset_ops.cc",
        "ops/igfs_ops.cc",
    ],
    deps = [
        ":dataset_kernels",
        ":igfs_kernels",
    ],
)

tf_gen_op_libs(
    op_lib_names = ["dataset_ops"],
)

tf_gen_op_libs(
    op_lib_names = ["igfs_ops"],
    deps = [":igfs_kernels"],
)

cc_library(
    name = "ignite_client",
    srcs = [
        "kernels/client/ignite_client.h",
        "kernels/client/ignite_byte_swapper.h",
        "kernels/client/ignite_plain_client.h",
        "kernels/client/ignite_ssl_wrapper.h",
        "kernels/client/ignite_ssl_wrapper.cc",
    ] + if_not_windows([
        "kernels/client/ignite_plain_client_unix.cc",
    ]) + if_windows([
        "kernels/client/ignite_plain_client_windows.cc",
    ]),
    copts = if_windows([
        "-DWIN32_LEAN_AND_MEAN",
    ]),
    deps = [
        "//tensorflow/core:framework_headers_lib",
        "@boringssl//:ssl",
        "@protobuf_archive//:protobuf_headers",
    ],
)

cc_library(
    name = "dataset_kernels",
    srcs = [
        "kernels/dataset/ignite_binary_object_parser.cc",
        "kernels/dataset/ignite_binary_object_parser.h",
        "kernels/dataset/ignite_dataset.cc",
        "kernels/dataset/ignite_dataset.h",
        "kernels/dataset/ignite_dataset_iterator.cc",
        "kernels/dataset/ignite_dataset_iterator.h",
        "kernels/dataset/ignite_dataset_ops.cc",
    ],
    deps = [
        ":ignite_client",
        "//tensorflow/core:framework_headers_lib",
        "//third_party/eigen3",
        "@protobuf_archive//:protobuf_headers",
    ],
    alwayslink = 1,
)

cc_library(
    name = "igfs_kernels",
    srcs = [
        "kernels/igfs/igfs.cc",
        "kernels/igfs/igfs.h",
        "kernels/igfs/igfs_client.cc",
        "kernels/igfs/igfs_client.h",
        "kernels/igfs/igfs_extended_tcp_client.cc",
        "kernels/igfs/igfs_extended_tcp_client.h",
        "kernels/igfs/igfs_messages.cc",
        "kernels/igfs/igfs_messages.h",
        "kernels/igfs/igfs_random_access_file.cc",
        "kernels/igfs/igfs_random_access_file.h",
        "kernels/igfs/igfs_writable_file.cc",
        "kernels/igfs/igfs_writable_file.h",
    ],
    deps = [":ignite_client"],
    alwayslink = 1,
)

py_library(
    name = "dataset_ops",
    srcs = [
        "python/ops/ignite_dataset_ops.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":ignite_op_loader",
        "//tensorflow/python:dataset_ops_gen",
        "//tensorflow/python:util",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/util:nest",
    ],
)

py_library(
    name = "igfs_ops",
    srcs = [
        "python/ops/igfs_ops.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":igfs_op_loader",
        "//tensorflow/python:util",
        "//tensorflow/python/data/util:nest",
    ],
)

tf_gen_op_wrapper_py(
    name = "gen_dataset_ops",
    out = "python/ops/gen_dataset_ops.py",
    deps = [":dataset_ops_op_lib"],
)

tf_gen_op_wrapper_py(
    name = "gen_igfs_ops",
    out = "python/ops/gen_igfs_ops.py",
    deps = [":igfs_ops_op_lib"],
)

tf_kernel_library(
    name = "dataset_ops_kernels",
    deps = [
        ":dataset_kernels",
        "//tensorflow/core:framework",
    ],
    alwayslink = 1,
)

tf_kernel_library(
    name = "igfs_ops_kernels",
    deps = [
        ":igfs_kernels",
        "//tensorflow/core:framework",
    ],
    alwayslink = 1,
)

tf_custom_op_py_library(
    name = "ignite_op_loader",
    srcs = ["python/ops/ignite_op_loader.py"],
    dso = [":_ignite_ops.so"],
    kernels = [
        ":dataset_ops_kernels",
        ":dataset_ops_op_lib",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":gen_dataset_ops",
        "//tensorflow/contrib/util:util_py",
        "//tensorflow/python:platform",
    ],
)

tf_custom_op_py_library(
    name = "igfs_op_loader",
    srcs = ["python/ops/igfs_op_loader.py"],
    dso = [":_ignite_ops.so"],
    kernels = [
        ":igfs_ops_kernels",
        ":igfs_ops_op_lib",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":gen_igfs_ops",
        "//tensorflow/contrib/util:util_py",
        "//tensorflow/python:platform",
    ],
)

# The Apache Ignite servers have to setup before the test and tear down
# after the test manually. The docker engine has to be installed.
#
# To setup Apache Ignite servers:
# $ bash ./python/tests/start_ignite.sh
#
# To tear down Apache Ignite servers:
# $ bash ./python/tests/stop_ignite.sh
tf_py_test(
    name = "ignite_test",
    srcs = [
        "python/tests/igfs_test.py",
        "python/tests/ignite_dataset_test.py",
    ],
    additional_deps = [
        ":ignite",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
    tags = [
        "manual",
        "no_windows",
        "notap",
    ],
)

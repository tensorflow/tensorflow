package(default_visibility = ["//tensorflow:internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load(
    "//tensorflow:tensorflow.bzl",
    "tf_gen_op_wrapper_py",
    "tf_kernel_library",
    "tf_custom_op_library",
    "tf_custom_op_py_library",
    "tf_gen_op_libs",
    "tf_py_test",
    "tf_cc_shared_object",
)

py_library(
    name = "igfs",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":igfs_ops",
    ],
)

tf_custom_op_library(
    name = "_igfs_ops.so",
    srcs = [
        "ops/igfs_ops.cc",
        "kernels/igfs.h",
    ],
    deps = [":igfs_kernels"],
)

tf_gen_op_libs(
    op_lib_names = ["igfs_ops"],
    deps = [":igfs_kernels"]
)

cc_library(
    name = "igfs_kernels",
    srcs = [
        "kernels/igfs.h",
        "kernels/igfs.cc",
        "kernels/igfs_random_access_file.h",
        "kernels/igfs_random_access_file.cc",
        "kernels/igfs_writable_file.h",
        "kernels/igfs_writable_file.cc",
        "kernels/igfs_client.h",
        "kernels/igfs_client.cc",
        "kernels/igfs_messages.h",
        "kernels/igfs_messages.cc",
        "kernels/igfs_extended_tcp_client.h",
        "kernels/igfs_extended_tcp_client.cc"
    ],
    deps = [
        "//tensorflow/contrib/ignite:ignite_client",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal"
    ],
    alwayslink=1,
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
    name = "gen_igfs_ops",
    out = "python/ops/gen_igfs_ops.py",
    deps = [":igfs_ops_op_lib"],
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
    name = "igfs_op_loader",
    srcs = ["python/ops/igfs_op_loader.py"],
    dso = [":_igfs_ops.so"],
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
    name = "igfs_test_py",
    srcs = ["python/tests/igfs_test.py"],
    additional_deps = [
        ":igfs",
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

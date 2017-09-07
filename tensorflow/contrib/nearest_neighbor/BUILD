# Description:
#   Tensorflow ops for nearest neighbor queries etc.

package(default_visibility = ["//tensorflow:__subpackages__"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "tf_custom_op_py_library")
load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
    "tf_custom_op_library",
    "tf_gen_op_libs",
    "tf_gen_op_wrapper_py",
    "tf_kernel_library",
    "tf_py_test",
)

tf_custom_op_library(
    name = "python/ops/_nearest_neighbor_ops.so",
    srcs = [
        "kernels/hyperplane_lsh_probes.cc",
        "ops/nearest_neighbor_ops.cc",
    ],
    deps = [
        ":hyperplane_lsh_probes",
    ],
)

tf_gen_op_libs(
    op_lib_names = ["nearest_neighbor_ops"],
)

tf_gen_op_wrapper_py(
    name = "nearest_neighbor_ops_pywrapper",
    deps = ["nearest_neighbor_ops_op_lib"],
)

tf_custom_op_py_library(
    name = "nearest_neighbor_py",
    srcs = ["__init__.py"] + glob(["python/ops/*.py"]),
    dso = [
        ":python/ops/_nearest_neighbor_ops.so",
    ],
    kernels = [
        ":nearest_neighbor_ops_kernels",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":nearest_neighbor_ops_pywrapper",
        "//tensorflow/contrib/util:util_py",
        "//tensorflow/python:framework_for_generated_wrappers",
    ],
)

tf_kernel_library(
    name = "nearest_neighbor_ops_kernels",
    srcs = ["kernels/hyperplane_lsh_probes.cc"],
    deps = [
        ":hyperplane_lsh_probes",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//third_party/eigen3",
    ],
)

cc_library(
    name = "heap",
    hdrs = [
        "kernels/heap.h",
    ],
)

tf_cc_test(
    name = "heap_test",
    size = "small",
    srcs = ["kernels/heap_test.cc"],
    deps = [
        ":heap",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

cc_library(
    name = "hyperplane_lsh_probes",
    hdrs = [
        "kernels/hyperplane_lsh_probes.h",
    ],
    deps = [
        ":heap",
        "//third_party/eigen3",
    ],
)

tf_cc_test(
    name = "hyperplane_lsh_probes_test_cc",
    size = "small",
    srcs = ["kernels/hyperplane_lsh_probes_test.cc"],
    deps = [
        ":hyperplane_lsh_probes",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_py_test(
    name = "hyperplane_lsh_probes_test",
    size = "small",
    srcs = ["python/kernel_tests/hyperplane_lsh_probes_test.py"],
    additional_deps = [
        ":nearest_neighbor_py",
        "//tensorflow/python:client_testlib",
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

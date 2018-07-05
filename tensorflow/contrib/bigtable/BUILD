# Cloud Bigtable client for TensorFlow

package(
    default_visibility = ["//tensorflow:internal"],
)

licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "tf_custom_op_py_library")
load(
    "//tensorflow:tensorflow.bzl",
    "tf_copts",
    "tf_custom_op_library",
    "tf_gen_op_libs",
    "tf_gen_op_wrapper_py",
    "tf_kernel_library",
    "tf_cc_test",
    "tf_py_test",
)

tf_custom_op_py_library(
    name = "bigtable",
    srcs = ["__init__.py"] + glob(["python/ops/*.py"]),
    dso = [
        ":python/ops/_bigtable.so",
    ],
    kernels = [
        ":bigtable_kernels",
        ":bigtable_ops_op_lib",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":bigtable_ops",
        "//tensorflow/contrib/util:util_py",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:platform",
        "//tensorflow/python:util",
        "//tensorflow/python/data",
    ],
)

KERNEL_FILES = [
    "kernels/bigtable_kernels.cc",
    "kernels/bigtable_lookup_dataset_op.cc",
    "kernels/bigtable_prefix_key_dataset_op.cc",
    "kernels/bigtable_range_key_dataset_op.cc",
    "kernels/bigtable_sample_keys_dataset_op.cc",
    "kernels/bigtable_scan_dataset_op.cc",
]

tf_custom_op_library(
    name = "python/ops/_bigtable.so",
    srcs = KERNEL_FILES + [
        "ops/bigtable_ops.cc",
    ],
    deps = [
        ":bigtable_lib_cc",
        "@com_github_googlecloudplatform_google_cloud_cpp//google/cloud/bigtable:bigtable_client",
    ],
)

tf_gen_op_wrapper_py(
    name = "bigtable_ops",
    deps = [":bigtable_ops_op_lib"],
)

tf_gen_op_libs(
    op_lib_names = [
        "bigtable_ops",
        "bigtable_test_ops",
    ],
)

tf_kernel_library(
    name = "bigtable_kernels",
    srcs = KERNEL_FILES,
    deps = [
        ":bigtable_lib_cc",
        "//tensorflow/core:framework_headers_lib",
        "//third_party/eigen3",
        "@com_github_googlecloudplatform_google_cloud_cpp//google/cloud/bigtable:bigtable_client",
    ],
)

# A library for use in the bigtable kernels.
cc_library(
    name = "bigtable_lib_cc",
    srcs = ["kernels/bigtable_lib.cc"],
    hdrs = ["kernels/bigtable_lib.h"],
    deps = [
        "//tensorflow/core:framework_headers_lib",
        "//third_party/eigen3",
        "@com_github_googlecloudplatform_google_cloud_cpp//google/cloud/bigtable:bigtable_client",
    ],
)

cc_library(
    name = "bigtable_test_client",
    srcs = ["kernels/test_kernels/bigtable_test_client.cc"],
    hdrs = ["kernels/test_kernels/bigtable_test_client.h"],
    deps = [
        "//tensorflow/core:framework_headers_lib",
        "@com_github_googleapis_googleapis//:bigtable_protos",
        "@com_github_googlecloudplatform_google_cloud_cpp//google/cloud/bigtable:bigtable_client",
        "@com_googlesource_code_re2//:re2",
    ],
)

tf_cc_test(
    name = "bigtable_test_client_test",
    srcs = ["kernels/test_kernels/bigtable_test_client_test.cc"],
    tags = ["manual"],
    deps = [
        ":bigtable_test_client",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@com_github_googlecloudplatform_google_cloud_cpp//google/cloud/bigtable:bigtable_client",
    ],
)

tf_gen_op_wrapper_py(
    name = "bigtable_test_ops",
    deps = [":bigtable_test_ops_op_lib"],
)

tf_custom_op_library(
    name = "python/kernel_tests/_bigtable_test.so",
    srcs = [
        "kernels/test_kernels/bigtable_test_client_op.cc",
        "ops/bigtable_test_ops.cc",
    ],
    deps = [
        ":bigtable_lib_cc",
        ":bigtable_test_client",
        "@com_googlesource_code_re2//:re2",
    ],
)

# Don't use tf_kernel_library because it prevents access to strings/stringprintf.h
cc_library(
    name = "bigtable_test_kernels",
    srcs = [
        "kernels/test_kernels/bigtable_test_client_op.cc",
    ],
    copts = tf_copts(),
    linkstatic = 1,
    deps = [
        ":bigtable_lib_cc",
        ":bigtable_test_client",
        "//tensorflow/core:framework_headers_lib",
        "//third_party/eigen3",
        "@com_googlesource_code_re2//:re2",
    ],
    alwayslink = 1,
)

tf_custom_op_py_library(
    name = "bigtable_test_py",
    dso = [
        ":python/kernel_tests/_bigtable_test.so",
    ],
    kernels = [
        ":bigtable_test_kernels",
        ":bigtable_test_ops_op_lib",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":bigtable_test_ops",
        # "//tensorflow/contrib/util:util_py",
        # "//tensorflow/python:framework_for_generated_wrappers",
        # "//tensorflow/python:platform",
        # "//tensorflow/python:util",
        # "//tensorflow/python/data",
    ],
)

tf_py_test(
    name = "bigtable_ops_test",
    size = "small",
    srcs = ["python/kernel_tests/bigtable_ops_test.py"],
    additional_deps = [
        ":bigtable",
        ":bigtable_test_py",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/contrib/util:util_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:platform",
        "//tensorflow/python:util",
    ],
    tags = ["manual"],
)

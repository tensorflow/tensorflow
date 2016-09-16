# Description:
# C API for TensorFlow, for use by client language bindings.

licenses(["notice"])  # Apache 2.0

load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
    "tf_cuda_library",
    "tf_custom_op_library",
)

# For platform specific build config
load(
    "//tensorflow/core:platform/default/build_config.bzl",
    "tf_kernel_tests_linkstatic",
)

# -----------------------------------------------------------------------------
# Public targets

tf_cuda_library(
    name = "c_api",
    srcs = ["c_api.cc"],
    hdrs = ["c_api.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
)

tf_cuda_library(
    name = "tf_status_helper",
    srcs = ["tf_status_helper.cc"],
    hdrs = ["tf_status_helper.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":c_api",
        "//tensorflow/core:lib",
    ],
)

tf_cuda_library(
    name = "checkpoint_reader",
    srcs = ["checkpoint_reader.cc"],
    hdrs = ["checkpoint_reader.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":tf_status_helper",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
)

# -----------------------------------------------------------------------------
# Tests

tf_cc_test(
    name = "c_api_test",
    size = "small",
    srcs = ["c_api_test.cc"],
    data = [":test_op.so"],
    linkopts = select({
        "//tensorflow:darwin": ["-headerpad_max_install_names"],
        "//conditions:default": [],
    }),
    linkstatic = tf_kernel_tests_linkstatic(),
    deps = [
        ":c_api",
        "//tensorflow/core:direct_session",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:proto_text",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels:array",
        "//tensorflow/core/kernels:math",
        "//third_party/eigen3",
    ],
)

tf_custom_op_library(
    name = "test_op.so",
    srcs = ["test_op.cc"],
)

# -----------------------------------------------------------------------------
# Google-internal targets.

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

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//visibility:public"])

load(
    "//tensorflow:tensorflow.bzl",
    "tf_gen_op_libs",
    "tf_py_test",
    "tf_custom_op_library",
    "tf_gen_op_wrapper_py",
)

cc_library(
    name = "all_ops",
    srcs = [":custom_op_sources"],
    hdrs = [":custom_op_headers"],
    deps = [
        "//tensorflow/core:framework_headers_lib",
    ],
    alwayslink = 1,
)

tf_custom_op_library(
    name = "python/ops/_periodic_resample_op.so",
    srcs = [
        ":custom_op_headers", 
        ":custom_op_sources",
    ],
)

py_library(
    name = "init_py",
    srcs = [
        "__init__.py",
        "python/__init__.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":periodic_resample_op_py",
    ],
)

py_library(
    name = "periodic_resample_op_py",
    srcs = ["python/ops/periodic_resample_op.py"],
    data = ["python/ops/_periodic_resample_op.so"],
    srcs_version = "PY2AND3",
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

filegroup(
    name = "custom_op_sources",
    srcs = glob(
        [
            "core/ops/*.cc",
            "core/kernels/*.cc",
        ],
        exclude = [
            "core/ops/*_test.cc",
            "core/kernels/*_test.cc",
        ],
    ),
)

filegroup(
    name = "custom_op_headers",
    srcs = glob(
        [
            "core/kernels/*.h",
            "core/ops/*.h",
        ],
    ),
)

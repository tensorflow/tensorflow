# Description:
# Tensor bundle: a module to efficiently serialize and deserialize tensors.

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

# To be exported to tensorflow/core:android_srcs.
filegroup(
    name = "android_srcs",
    srcs = [
        "tensor_bundle.cc",
        "tensor_bundle.h",
    ],
)

cc_library(
    name = "tensor_bundle",
    srcs = ["tensor_bundle.cc"],
    hdrs = ["tensor_bundle.h"],
    deps = [
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:proto_text",
        "//tensorflow/core:protos_all_cc",
    ],
)

cc_test(
    name = "tensor_bundle_test",
    srcs = ["tensor_bundle_test.cc"],
    deps = [
        ":tensor_bundle",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensor_testutil",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

# -----------------------------------------------------------------------------
# Google-internal targets.  These must be at the end for syncrepo.

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

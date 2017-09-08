# Description:
#   Tensor bundle: a module to efficiently serialize and deserialize tensors.

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

load(
    "//tensorflow:tensorflow.bzl",
    "cc_header_only_library",
    "if_not_windows",
    "tf_copts",
)

# To be exported to tensorflow/core:mobile_srcs.
filegroup(
    name = "mobile_srcs",
    srcs = [
        "naming.cc",
        "naming.h",
        "tensor_bundle.cc",
        "tensor_bundle.h",
    ],
)

alias(
    name = "android_srcs",
    actual = ":mobile_srcs",
)

cc_library(
    name = "tensor_bundle",
    srcs = ["tensor_bundle.cc"],
    hdrs = ["tensor_bundle.h"],
    copts = tf_copts() + if_not_windows(["-Wno-sign-compare"]),
    deps = [
        ":naming",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:proto_text",
        "//tensorflow/core:protos_all_cc",
    ],
)

cc_header_only_library(
    name = "tensor_bundle_headers_lib",
    features = ["-parse_headers"],  # Transitively pulls in Eigen headers
    deps = [":tensor_bundle"],
)

cc_library(
    name = "naming",
    srcs = ["naming.cc"],
    hdrs = ["naming.h"],
    deps = ["//tensorflow/core:lib"],
)

cc_test(
    name = "tensor_bundle_test",
    size = "small",
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

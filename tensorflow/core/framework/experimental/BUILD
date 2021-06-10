# Experimental features in the TF framework.
load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
)
load(
    "//tensorflow/core/platform:rules_cc.bzl",
    "cc_library",
)
load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_kernel_tests_linkstatic",
)

package(
    default_visibility = [
        "//tensorflow/core:__subpackages__",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "type_inference",
    srcs = [
        "type_inference.cc",
    ],
    hdrs = ["type_inference.h"],
    deps = [
        "//tensorflow/core/framework:full_type_proto_cc",
        "//tensorflow/core/framework:full_type_util",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

tf_cc_test(
    name = "type_inference_test",
    srcs = [
        "type_inference.h",
        "type_inference_test.cc",
    ],
    linkstatic = tf_kernel_tests_linkstatic(),
    visibility = [
        "//tensorflow:internal",
        "//tensorflow/core:__pkg__",
    ],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

exports_files(
    srcs = [
        "type_inference.h",
    ],
    visibility = [
        "//tensorflow/core/framework:__pkg__",
    ],
)

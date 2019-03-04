package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "tf_cc_test")

cc_library(
    name = "decode",
    hdrs = ["decode.h"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "descriptors",
    srcs = ["descriptors.cc"],
    hdrs = ["descriptors.h"],
    deps = [
        ":descriptor_pool_registry",
        ":local_descriptor_pool_registration",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "descriptor_pool_registry",
    srcs = ["descriptor_pool_registry.cc"],
    hdrs = ["descriptor_pool_registry.h"],
    deps = [
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "descriptor_pool_registry_test",
    srcs = ["descriptor_pool_registry_test.cc"],
    deps = [
        ":descriptor_pool_registry",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

# Depending on this target adds support for using the special
# value "local://" (or "") for descriptor source, in which case
# descriptors linked into the code will be searched.
cc_library(
    name = "local_descriptor_pool_registration",
    srcs = ["local_descriptor_pool_registration.cc"],
    deps = [
        ":descriptor_pool_registry",
        "//tensorflow/core:lib",
    ],
    alwayslink = 1,
)

cc_library(
    name = "proto_utils",
    srcs = ["proto_utils.cc"],
    hdrs = ["proto_utils.h"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:platform_base",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "proto_utils_test",
    srcs = ["proto_utils_test.cc"],
    deps = [
        ":proto_utils",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "@com_google_googletest//:gtest_main",
    ],
)

load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//tensorflow:tensorflow.bzl", "tf_cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

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
        "@com_google_absl//absl/strings",
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
        "//tensorflow/tsl/util/proto:proto_utils",
        "@com_google_absl//absl/strings",
    ],
)

# For a more maintainable build this target should not exist and the headers
# should  be split into the existing cc_library targets, but this change was
# automatically  done so that we can remove long standing issues and complexity
# in the build system. It's up to the OWNERS of this package to get rid of it or
# not. The use of the textual_hdrs attribute is discouraged, use hdrs instead.
# Here it is used to avoid header parsing errors in packages where the feature
# parse_headers was enabled since loose headers were not being parsed. See
# go/loose-lsc-one-target-approach for more details.
cc_library(
    name = "loose_headers",
    tags = ["avoid_dep"],
    textual_hdrs = ["proto_utils.h"],
    visibility = ["//tensorflow/contrib/fused_conv:__pkg__"],
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
        "@com_google_googletest//:gtest",
    ],
)

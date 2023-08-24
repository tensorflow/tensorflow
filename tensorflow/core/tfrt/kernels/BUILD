load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":internal"],
)

package_group(
    name = "internal",
    packages = [
        "//tensorflow/core/tfrt/...",
    ],
)

cc_library(
    name = "stream_ops",
    srcs = ["stream_ops.cc"],
    hdrs = ["stream_ops.h"],
    deps = [
        ":stream_ops_util",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/tfrt/runtime:stream",
        "//tensorflow/tsl/profiler/lib:traceme",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
    ],
    alwayslink = 1,
)

cc_library(
    name = "stream_ops_util_constants",
    hdrs = ["stream_ops_util_constants.h"],
    visibility = [
        "//visibility:public",
    ],
)

cc_library(
    name = "stream_ops_util",
    srcs = ["stream_ops_util.cc"],
    hdrs = ["stream_ops_util.h"],
    deps = [
        ":stream_ops_util_constants",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_types_hdr",
        "//tensorflow/core/framework:types_proto_cc",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

tf_cc_test(
    name = "stream_ops_util_test",
    srcs = ["stream_ops_util_test.cc"],
    tags = ["no_oss"],
    deps = [
        ":stream_ops_util",
        "//tensorflow/core:framework",
        "//tensorflow/core/framework:tensor_matcher",
        "//tensorflow/core/framework:tensor_testutil",
        "@com_google_googletest//:gtest_main",
    ],
)

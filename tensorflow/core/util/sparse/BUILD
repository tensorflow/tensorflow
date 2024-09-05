load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("//tensorflow:tensorflow.default.bzl", "filegroup")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

filegroup(
    name = "higher_level_tests_group",
    srcs = ["sparse_tensor_test.cc"],
    visibility = ["//visibility:private"],
)

# Export source files needed for mobile builds, which do not use granular targets.
filegroup(
    name = "mobile_srcs_only_runtime",
    srcs = [
        "dim_comparator.h",
        "group_iterator.cc",
        "group_iterator.h",
        "sparse_tensor.cc",
        "sparse_tensor.h",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "framework_group",
    srcs = [
        "group_iterator.h",
        "sparse_tensor.h",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "framework_internal_private_headers_group",
    srcs = [
        "dim_comparator.h",
        "group_iterator.h",
        "sparse_tensor.h",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "framework_internal_impl_group",
    srcs = [
        "group_iterator.cc",
        "sparse_tensor.cc",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

cc_library(
    name = "sparse_tensor",
    srcs = ["sparse_tensor.cc"],
    hdrs = [
        "dim_comparator.h",
        "sparse_tensor.h",
    ],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_lite",
        "//tensorflow/core:framework_types_hdr",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_proto_parsing",
        "//tensorflow/core:portable_gif_internal",
        "//tensorflow/core/framework:bounds_check",
        "//tensorflow/core/framework:types_proto_cc",
        "//tensorflow/core/lib/core:errors",
        "//tensorflow/core/lib/core:status",
        "//tensorflow/core/platform:errors",
        "@com_google_absl//absl/base:core_headers",
        "@eigen_archive//:eigen3",
    ],
)

tf_cc_test(
    name = "sparse_tensor_test",
    size = "small",
    srcs = [
        "sparse_tensor_test.cc",
    ],
    deps = [
        ":sparse_tensor",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_lite",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core/platform:statusor",
        "@com_google_googletest//:gtest_main",
        "@eigen_archive//:eigen3",
    ],
)

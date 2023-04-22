load("//tensorflow:tensorflow.bzl", "filegroup")

package(
    licenses = ["notice"],
)

filegroup(
    name = "higher_level_tests_group",
    srcs = ["sparse_tensor_test.cc"],
    visibility = ["//tensorflow/core:__pkg__"],
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

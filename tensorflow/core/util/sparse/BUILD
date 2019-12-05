package(
    licenses = ["notice"],  # Apache 2.0
)

filegroup(
    name = "higher_level_tests_group",
    srcs = ["sparse_tensor_test.cc"],
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "mobile_srcs_no_runtime_group",
    srcs = [
        "dim_comparator.h",
        "group_iterator.cc",
        "group_iterator.h",
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
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")

cc_library(
    name = "list_ops_util",
    srcs = ["list_ops_util.cc"],
    hdrs = ["list_ops_util.h"],
    deps = [
        "//tensorflow/lite:util",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core/c:common",
    ],
)

cc_test(
    name = "list_ops_util_test",
    srcs = ["list_ops_util_test.cc"],
    deps = [
        ":list_ops_util",
        "//tensorflow/lite:util",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest_main",
    ],
)

# test utilities for complex multi-op subgraphs
cc_library(
    name = "list_ops_subgraph_test_util",
    testonly = True,
    srcs = ["list_ops_subgraph_test_util.cc"],
    hdrs = ["list_ops_subgraph_test_util.h"],
    deps = [
        "//tensorflow/lite:builtin_ops",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core:cc_api_stable",
        "//tensorflow/lite/core:subgraph",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/kernels:op_macros",
        "//tensorflow/lite/kernels:subgraph_test_util",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "list_ops_subgraph_test",
    srcs = ["list_ops_subgraph_test.cc"],
    deps = [
        ":list_ops_subgraph_test_util",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest_main",
    ],
)

# type erased class that sits behind the `data.data` of variant tensors
cc_library(
    name = "tensor_array",
    srcs = ["tensor_array.cc"],
    hdrs = ["tensor_array.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/lite:util",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core/c:common",
    ],
)

cc_test(
    name = "tensor_array_test",
    srcs = ["tensor_array_test.cc"],
    deps = [
        ":tensor_array",
        "//tensorflow/lite:type_to_tflitetype",
        "//tensorflow/lite:util",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core/c:c_api_types",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest_main",
    ],
)

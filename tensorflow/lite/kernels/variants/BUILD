load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")

# Library of list ops.
cc_library(
    name = "list_ops_lib",
    srcs = glob(
        ["list_kernels/*.cc"],
        exclude = ["list_kernels/*test*"],
    ),
    hdrs = ["list_ops_lib.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":list_ops_util",
        ":tensor_array",
        "//tensorflow/lite:array",
        "//tensorflow/lite:util",
        "//tensorflow/lite/core/c:c_api_types",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/kernels:kernel_util",
        "//tensorflow/lite/kernels/internal:tensor_ctypes",
        "//tensorflow/lite/schema:schema_fbs",
    ],
)

# Utils for simple kernel unit tests.
cc_library(
    name = "test_util",
    testonly = True,
    srcs = ["list_kernels/test_util.cc"],
    hdrs = ["list_kernels/test_util.h"],
    deps = [
        ":tensor_array",
        "//tensorflow/lite:array",
        "//tensorflow/lite:type_to_tflitetype",
        "//tensorflow/lite:util",
        "//tensorflow/lite/core/c:c_api_types",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/kernels:kernel_util",
        "//tensorflow/lite/kernels:op_macros",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "list_reserve_test",
    srcs = ["list_kernels/list_reserve_test.cc"],
    deps = [
        ":list_ops_lib",
        ":tensor_array",
        "//tensorflow/lite/core/c:c_api_types",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/kernels:test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "list_stack_test",
    srcs = ["list_kernels/list_stack_test.cc"],
    deps = [
        ":list_ops_lib",
        ":test_util",
        "//tensorflow/lite:util",
        "//tensorflow/lite/core/c:c_api_types",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/kernels:test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "list_set_item_test",
    srcs = ["list_kernels/list_set_item_test.cc"],
    deps = [
        ":list_ops_lib",
        ":tensor_array",
        ":test_util",
        "//tensorflow/lite:type_to_tflitetype",
        "//tensorflow/lite/core/c:c_api_types",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/kernels:test_util",
        "//tensorflow/lite/kernels/internal:compatibility",
        "//tensorflow/lite/schema:schema_fbs",
        "//third_party/tflite_micro/tensorflow/lite/micro:memory_helpers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "list_from_tensor_test",
    srcs = ["list_kernels/list_from_tensor_test.cc"],
    deps = [
        ":list_ops_lib",
        ":tensor_array",
        "//tensorflow/lite/core/c:c_api_types",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/kernels:test_util",
        "//tensorflow/lite/kernels/internal:compatibility",
        "//tensorflow/lite/kernels/internal:tensor_ctypes",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "list_ops_util",
    srcs = ["list_ops_util.cc"],
    hdrs = ["list_ops_util.h"],
    deps = [
        "//tensorflow/lite:util",
        "//tensorflow/lite/core/c:common",
    ],
)

cc_test(
    name = "list_ops_util_test",
    srcs = ["list_ops_util_test.cc"],
    deps = [
        ":list_ops_util",
        "//tensorflow/lite:util",
        "//tensorflow/lite/core/c:c_api_types",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest_main",
    ],
)

# Utils for testing complex multi-op subgraphs.
cc_library(
    name = "list_ops_subgraph_test_util",
    testonly = True,
    srcs = ["list_ops_subgraph_test_util.cc"],
    hdrs = ["list_ops_subgraph_test_util.h"],
    deps = [
        ":list_ops_lib",
        "//tensorflow/lite:builtin_ops",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core:cc_api_stable",
        "//tensorflow/lite/core:subgraph",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/kernels:op_macros",
        "//tensorflow/lite/kernels:subgraph_test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "list_ops_subgraph_test",
    srcs = ["list_ops_subgraph_test.cc"],
    deps = [
        ":list_ops_subgraph_test_util",
        ":tensor_array",
        "//tensorflow/lite:interpreter_test_util",
        "//tensorflow/lite:util",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels:kernel_util",
        "//tensorflow/lite/kernels:test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

# Type erased class that sits behind the `data.data` of variant tensors.
cc_library(
    name = "tensor_array",
    srcs = ["tensor_array.cc"],
    hdrs = ["tensor_array.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/lite:util",
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

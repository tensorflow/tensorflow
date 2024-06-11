load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load("//tensorflow/lite:build_def.bzl", "tflite_copts", "tflite_copts_warnings", "tflite_linkopts")

common_copts = tflite_copts() + tflite_copts_warnings()

# Library of list ops.
cc_library(
    name = "list_ops_lib",
    srcs = glob(
        ["list_kernels/*.cc"],
        exclude = ["list_kernels/*test*"],
    ) + ["register_list_ops.cc"],
    hdrs = [
        "list_ops_lib.h",
        "register_list_ops.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":list_ops_util",
        ":tensor_array",
        "//tensorflow/lite:array",
        "//tensorflow/lite:mutable_op_resolver",
        "//tensorflow/lite:util",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/core/c:c_api_types",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/kernels:cpu_backend_context",
        "//tensorflow/lite/kernels:kernel_util",
        "//tensorflow/lite/kernels/internal:optimized_base",
        "//tensorflow/lite/kernels/internal:runtime_shape",
        "//tensorflow/lite/kernels/internal:tensor",
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
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core/c:c_api_types",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/kernels:kernel_util",
        "//tensorflow/lite/kernels:op_macros",
        "//tensorflow/lite/kernels:test_util",
        "//tensorflow/lite/schema:schema_fbs",
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
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "list_from_tensor_test",
    srcs = ["list_kernels/list_from_tensor_test.cc"],
    deps = [
        ":list_ops_lib",
        ":tensor_array",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core/c:c_api_types",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/kernels:test_util",
        "//tensorflow/lite/kernels/internal:compatibility",
        "//tensorflow/lite/kernels/internal:tensor_ctypes",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "list_get_item_test",
    srcs = ["list_kernels/list_get_item_test.cc"],
    deps = [
        ":list_ops_lib",
        ":test_util",
        "//tensorflow/lite/core/c:c_api_types",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/kernels:test_util",
        "//tensorflow/lite/kernels/internal:tensor_ctypes",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "list_length_test",
    srcs = ["list_kernels/list_length_test.cc"],
    deps = [
        ":list_ops_lib",
        ":tensor_array",
        ":test_util",
        "//tensorflow/lite/core/c:c_api_types",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/kernels:test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "list_element_shape_test",
    srcs = ["list_kernels/list_element_shape_test.cc"],
    deps = [
        ":list_ops_lib",
        ":tensor_array",
        ":test_util",
        "//tensorflow/lite/core/c:c_api_types",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/kernels:test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "list_pop_back_test",
    srcs = ["list_kernels/list_pop_back_test.cc"],
    deps = [
        ":list_ops_lib",
        ":tensor_array",
        ":test_util",
        "//tensorflow/lite/core/c:c_api_types",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/kernels:test_util",
        "//tensorflow/lite/kernels/internal:tensor_ctypes",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "list_push_back_test",
    srcs = ["list_kernels/list_push_back_test.cc"],
    deps = [
        ":list_ops_lib",
        ":tensor_array",
        ":test_util",
        "//tensorflow/lite:type_to_tflitetype",
        "//tensorflow/lite/core/c:c_api_types",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/kernels:kernel_util",
        "//tensorflow/lite/kernels:test_util",
        "//tensorflow/lite/kernels/internal:compatibility",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "variant_add_n_test",
    srcs = ["list_kernels/variant_add_n_test.cc"],
    deps = [
        ":list_ops_lib",
        ":tensor_array",
        ":test_util",
        "//tensorflow/lite:type_to_tflitetype",
        "//tensorflow/lite/core/c:c_api_types",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/kernels:kernel_util",
        "//tensorflow/lite/kernels:test_util",
        "//tensorflow/lite/kernels/internal:compatibility",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "variant_zeros_like_test",
    srcs = ["list_kernels/variant_zeros_like_test.cc"],
    deps = [
        ":list_ops_lib",
        ":tensor_array",
        ":test_util",
        "//tensorflow/lite/core/c:c_api_types",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/kernels:kernel_util",
        "//tensorflow/lite/kernels:test_util",
        "//tensorflow/lite/kernels/internal:compatibility",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "list_ops_util",
    srcs = ["list_ops_util.cc"],
    hdrs = ["list_ops_util.h"],
    visibility = ["//tensorflow/lite/kernels/variants:__subpackages__"],
    deps = [
        ":tensor_array",
        "//tensorflow/lite:array",
        "//tensorflow/lite:util",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/core/c:common",
    ],
)

cc_test(
    name = "list_ops_util_test",
    srcs = ["list_ops_util_test.cc"],
    deps = [
        ":list_ops_util",
        ":tensor_array",
        "//tensorflow/lite:array",
        "//tensorflow/lite:util",
        "//tensorflow/lite/core/c:c_api_types",
        "//tensorflow/lite/core/c:common",
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
    visibility = ["//tensorflow/lite:__subpackages__"],
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

# Benchmark tool override with registered custom tensorlist ops.
cc_library(
    name = "register_list_ops_for_benchmark",
    srcs = ["register_list_ops.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":list_ops_lib",
        "//tensorflow/lite:mutable_op_resolver",
        "//tensorflow/lite:op_resolver",
        "//tensorflow/lite/tools/benchmark:register_custom_op",
    ],
    alwayslink = 1,
)

cc_binary(
    name = "benchmark_model_with_list_ops",
    copts = common_copts,
    linkopts = tflite_linkopts() + select({
        "//tensorflow:android": [
            "-pie",  # Android 5.0 and later supports only PIE
            "-lm",  # some builtin ops, e.g., tanh, need -lm
            "-Wl,--rpath=/data/local/tmp/",  # Hexagon delegate libraries should be in /data/local/tmp
        ],
        "//conditions:default": [],
    }),
    tags = ["builder_default_android_arm64"],
    deps = [
        ":register_list_ops_for_benchmark",
        "//tensorflow/lite/tools/benchmark:benchmark_model_main",
    ],
)

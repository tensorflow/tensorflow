package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "tfl_tensor_ref",
    srcs = ["tfl_tensor_ref.cc"],
    hdrs = ["tfl_tensor_ref.h"],
    deps = [
        "//tensorflow/lite:array",
        "//tensorflow/lite/core/c:c_api_types",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/experimental/ml_adjacent:lib",
        "//tensorflow/lite/kernels:kernel_util",
        "//tensorflow/lite/kernels/internal:compatibility",
    ],
)

cc_test(
    name = "tfl_tensor_ref_test",
    srcs = ["tfl_tensor_ref_test.cc"],
    deps = [
        ":tfl_tensor_ref",
        "//tensorflow/lite:util",
        "//tensorflow/lite/core/c:c_api_types",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/experimental/ml_adjacent:lib",
        "//tensorflow/lite/kernels:kernel_util",
        "//tensorflow/lite/kernels:test_util",
        "//tensorflow/lite/kernels/internal:compatibility",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "extern_call",
    srcs = ["extern_call.cc"],
    hdrs = ["extern_call.h"],
    deps = [
        ":tfl_tensor_ref",
        "//tensorflow/lite/core/c:c_api_types",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/experimental/ml_adjacent:lib",
        "//tensorflow/lite/experimental/ml_adjacent/algo:crop",
        "//tensorflow/lite/experimental/ml_adjacent/algo:resize",
        "//tensorflow/lite/kernels:kernel_util",
    ],
)

cc_test(
    name = "extern_call_test",
    srcs = ["extern_call_test.cc"],
    deps = [
        ":extern_call",
        "//tensorflow/lite/core/c:c_api_types",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/kernels:test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

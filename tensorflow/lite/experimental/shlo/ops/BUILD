load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

# Implementation of StableHLO operations.
load("//tensorflow/lite/experimental/shlo:build_def.bzl", "shlo_ref_linkopts")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:LICENSE"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "benchmark_util",
    testonly = True,
    hdrs = ["benchmark_util.h"],
    deps = [
        "//tensorflow/lite/experimental/shlo:bf16",
        "//tensorflow/lite/experimental/shlo:data_type",
        "//tensorflow/lite/experimental/shlo:f16",
    ],
)

cc_library(
    name = "is_finite",
    srcs = ["is_finite.cc"],
    hdrs = ["is_finite.h"],
    deps = [
        "//tensorflow/lite/experimental/shlo:data_type",
        "//tensorflow/lite/experimental/shlo:dispatch",
        "//tensorflow/lite/experimental/shlo:shape",
        "//tensorflow/lite/experimental/shlo:tensor",
        "@com_google_absl//absl/status",
    ],
)

cc_test(
    name = "is_finite_test",
    srcs = ["is_finite_test.cc"],
    linkopts = shlo_ref_linkopts(),
    deps = [
        ":is_finite",
        "//tensorflow/lite/experimental/shlo:bf16",
        "//tensorflow/lite/experimental/shlo:data_type",
        "//tensorflow/lite/experimental/shlo:f16",
        "//tensorflow/lite/experimental/shlo:shape",
        "//tensorflow/lite/experimental/shlo:status_matcher",
        "//tensorflow/lite/experimental/shlo:tensor",
        "//tensorflow/lite/experimental/shlo:tensor_matcher",
        "//tensorflow/lite/experimental/shlo:tensor_with_data",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "is_finite_bench",
    srcs = ["is_finite_bench.cc"],
    linkopts = shlo_ref_linkopts(),
    deps = [
        ":benchmark_util",
        ":is_finite",
        "//tensorflow/lite/experimental/shlo:data_type",
        "//tensorflow/lite/experimental/shlo:shape",
        "//tensorflow/lite/experimental/shlo:tensor",
        "//tensorflow/lite/experimental/shlo:tensor_with_data",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_benchmark//:benchmark",
    ],
)

cc_library(
    name = "util",
    srcs = ["util.cc"],
    hdrs = ["util.h"],
    deps = [
        "//tensorflow/lite/experimental/shlo:data_type",
        "//tensorflow/lite/experimental/shlo:shape",
        "//tensorflow/lite/experimental/shlo:tensor",
        "@com_google_absl//absl/status",
    ],
)

cc_test(
    name = "util_test",
    srcs = ["util_test.cc"],
    linkopts = shlo_ref_linkopts(),
    deps = [
        ":util",
        "//tensorflow/lite/experimental/shlo:shape",
        "//tensorflow/lite/experimental/shlo:status_matcher",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "unary_elementwise",
    hdrs = ["unary_elementwise.h"],
    deps = [
        ":util",
        "//tensorflow/lite/experimental/shlo:data_type",
        "//tensorflow/lite/experimental/shlo:dispatch",
        "//tensorflow/lite/experimental/shlo:quantize",
        "//tensorflow/lite/experimental/shlo:quantized_tensor_element_type",
        "//tensorflow/lite/experimental/shlo:shape",
        "//tensorflow/lite/experimental/shlo:tensor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "unary_elementwise_test",
    srcs = ["unary_elementwise_test.cc"],
    linkopts = shlo_ref_linkopts(),
    deps = [
        ":test_util",
        ":unary_elementwise",
        "//tensorflow/lite/experimental/shlo:data_type",
        "//tensorflow/lite/experimental/shlo:quantized_tensor_element_type",
        "//tensorflow/lite/experimental/shlo:shape",
        "//tensorflow/lite/experimental/shlo:status_matcher",
        "//tensorflow/lite/experimental/shlo:tensor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "test_util",
    testonly = True,
    hdrs = ["test_util.h"],
    deps = [
        "//tensorflow/lite/experimental/shlo:data_type",
        "//tensorflow/lite/experimental/shlo:i4",
        "//tensorflow/lite/experimental/shlo:quantized_tensor_element_type",
        "//tensorflow/lite/experimental/shlo:shape",
        "//tensorflow/lite/experimental/shlo:tensor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "unary_elementwise_test_util",
    testonly = True,
    hdrs = ["unary_elementwise_test_util.h"],
    deps = [
        ":test_util",
        "//tensorflow/lite/experimental/shlo:data_type",
        "//tensorflow/lite/experimental/shlo:shape",
        "//tensorflow/lite/experimental/shlo:status_matcher",
        "//tensorflow/lite/experimental/shlo:tensor",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "abs",
    srcs = ["abs.cc"],
    hdrs = ["abs.h"],
    deps = [
        ":unary_elementwise",
        ":util",
        "//tensorflow/lite/experimental/shlo:dispatch",
        "//tensorflow/lite/experimental/shlo:tensor",
        "@com_google_absl//absl/status",
    ],
)

cc_test(
    name = "abs_test",
    srcs = ["abs_test.cc"],
    linkopts = shlo_ref_linkopts(),
    deps = [
        ":abs",
        ":test_util",
        ":unary_elementwise_test_util",
        "//tensorflow/lite/experimental/shlo:quantize",
        "//tensorflow/lite/experimental/shlo:quantized_tensor_element_type",
        "//tensorflow/lite/experimental/shlo:shape",
        "//tensorflow/lite/experimental/shlo:status_matcher",
        "//tensorflow/lite/experimental/shlo:tensor",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "cbrt",
    srcs = ["cbrt.cc"],
    hdrs = ["cbrt.h"],
    deps = [
        ":unary_elementwise",
        ":util",
        "//tensorflow/lite/experimental/shlo:bf16",
        "//tensorflow/lite/experimental/shlo:dispatch",
        "//tensorflow/lite/experimental/shlo:f16",
        "//tensorflow/lite/experimental/shlo:tensor",
        "@com_google_absl//absl/status",
    ],
)

cc_test(
    name = "cbrt_test",
    srcs = ["cbrt_test.cc"],
    linkopts = shlo_ref_linkopts(),
    deps = [
        ":cbrt",
        ":test_util",
        ":unary_elementwise_test_util",
        "//tensorflow/lite/experimental/shlo:bf16",
        "//tensorflow/lite/experimental/shlo:f16",
        "//tensorflow/lite/experimental/shlo:quantize",
        "//tensorflow/lite/experimental/shlo:quantized_tensor_element_type",
        "//tensorflow/lite/experimental/shlo:shape",
        "//tensorflow/lite/experimental/shlo:status_matcher",
        "//tensorflow/lite/experimental/shlo:tensor",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "ceil",
    srcs = ["ceil.cc"],
    hdrs = ["ceil.h"],
    deps = [
        ":unary_elementwise",
        ":util",
        "//tensorflow/lite/experimental/shlo:bf16",
        "//tensorflow/lite/experimental/shlo:dispatch",
        "//tensorflow/lite/experimental/shlo:f16",
        "//tensorflow/lite/experimental/shlo:tensor",
        "@com_google_absl//absl/status",
    ],
)

cc_test(
    name = "ceil_test",
    srcs = ["ceil_test.cc"],
    linkopts = shlo_ref_linkopts(),
    deps = [
        ":ceil",
        ":test_util",
        ":unary_elementwise_test_util",
        "//tensorflow/lite/experimental/shlo:bf16",
        "//tensorflow/lite/experimental/shlo:f16",
        "//tensorflow/lite/experimental/shlo:quantize",
        "//tensorflow/lite/experimental/shlo:quantized_tensor_element_type",
        "//tensorflow/lite/experimental/shlo:shape",
        "//tensorflow/lite/experimental/shlo:status_matcher",
        "//tensorflow/lite/experimental/shlo:tensor",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "cosine",
    srcs = ["cosine.cc"],
    hdrs = ["cosine.h"],
    linkopts = shlo_ref_linkopts(),
    deps = [
        ":unary_elementwise",
        ":util",
        "//tensorflow/lite/experimental/shlo:bf16",
        "//tensorflow/lite/experimental/shlo:dispatch",
        "//tensorflow/lite/experimental/shlo:f16",
        "//tensorflow/lite/experimental/shlo:tensor",
        "@com_google_absl//absl/status",
    ],
)

cc_test(
    name = "cosine_test",
    srcs = ["cosine_test.cc"],
    deps = [
        ":cosine",
        ":test_util",
        ":unary_elementwise_test_util",
        "//tensorflow/lite/experimental/shlo:bf16",
        "//tensorflow/lite/experimental/shlo:f16",
        "//tensorflow/lite/experimental/shlo:quantize",
        "//tensorflow/lite/experimental/shlo:quantized_tensor_element_type",
        "//tensorflow/lite/experimental/shlo:shape",
        "//tensorflow/lite/experimental/shlo:status_matcher",
        "//tensorflow/lite/experimental/shlo:tensor",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "sine",
    srcs = ["sine.cc"],
    hdrs = ["sine.h"],
    deps = [
        ":unary_elementwise",
        ":util",
        "//tensorflow/lite/experimental/shlo:bf16",
        "//tensorflow/lite/experimental/shlo:data_type",
        "//tensorflow/lite/experimental/shlo:dispatch",
        "//tensorflow/lite/experimental/shlo:f16",
        "//tensorflow/lite/experimental/shlo:tensor",
        "@com_google_absl//absl/status",
    ],
)

cc_test(
    name = "sine_test",
    srcs = ["sine_test.cc"],
    linkopts = shlo_ref_linkopts(),
    deps = [
        ":sine",
        ":test_util",
        ":unary_elementwise_test_util",
        "//tensorflow/lite/experimental/shlo:bf16",
        "//tensorflow/lite/experimental/shlo:f16",
        "//tensorflow/lite/experimental/shlo:quantize",
        "//tensorflow/lite/experimental/shlo:quantized_tensor_element_type",
        "//tensorflow/lite/experimental/shlo:shape",
        "//tensorflow/lite/experimental/shlo:status_matcher",
        "//tensorflow/lite/experimental/shlo:tensor",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "tanh",
    srcs = ["tanh.cc"],
    hdrs = ["tanh.h"],
    deps = [
        ":unary_elementwise",
        ":util",
        "//tensorflow/lite/experimental/shlo:bf16",
        "//tensorflow/lite/experimental/shlo:data_type",
        "//tensorflow/lite/experimental/shlo:dispatch",
        "//tensorflow/lite/experimental/shlo:f16",
        "//tensorflow/lite/experimental/shlo:tensor",
        "@com_google_absl//absl/status",
    ],
)

cc_test(
    name = "tanh_test",
    srcs = ["tanh_test.cc"],
    linkopts = shlo_ref_linkopts(),
    deps = [
        ":tanh",
        ":test_util",
        ":unary_elementwise_test_util",
        "//tensorflow/lite/experimental/shlo:bf16",
        "//tensorflow/lite/experimental/shlo:f16",
        "//tensorflow/lite/experimental/shlo:quantize",
        "//tensorflow/lite/experimental/shlo:quantized_tensor_element_type",
        "//tensorflow/lite/experimental/shlo:shape",
        "//tensorflow/lite/experimental/shlo:status_matcher",
        "//tensorflow/lite/experimental/shlo:tensor",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "sqrt",
    srcs = ["sqrt.cc"],
    hdrs = ["sqrt.h"],
    deps = [
        ":unary_elementwise",
        ":util",
        "//tensorflow/lite/experimental/shlo:bf16",
        "//tensorflow/lite/experimental/shlo:dispatch",
        "//tensorflow/lite/experimental/shlo:f16",
        "//tensorflow/lite/experimental/shlo:tensor",
        "@com_google_absl//absl/status",
    ],
)

cc_test(
    name = "sqrt_test",
    srcs = ["sqrt_test.cc"],
    linkopts = shlo_ref_linkopts(),
    deps = [
        ":sqrt",
        ":test_util",
        ":unary_elementwise_test_util",
        "//tensorflow/lite/experimental/shlo:bf16",
        "//tensorflow/lite/experimental/shlo:f16",
        "//tensorflow/lite/experimental/shlo:quantize",
        "//tensorflow/lite/experimental/shlo:quantized_tensor_element_type",
        "//tensorflow/lite/experimental/shlo:shape",
        "//tensorflow/lite/experimental/shlo:status_matcher",
        "//tensorflow/lite/experimental/shlo:tensor",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "exponential",
    srcs = ["exponential.cc"],
    hdrs = ["exponential.h"],
    deps = [
        ":unary_elementwise",
        ":util",
        "//tensorflow/lite/experimental/shlo:bf16",
        "//tensorflow/lite/experimental/shlo:dispatch",
        "//tensorflow/lite/experimental/shlo:f16",
        "//tensorflow/lite/experimental/shlo:tensor",
        "@com_google_absl//absl/status",
    ],
)

cc_test(
    name = "exponential_test",
    srcs = ["exponential_test.cc"],
    linkopts = shlo_ref_linkopts(),
    deps = [
        ":exponential",
        ":test_util",
        ":unary_elementwise_test_util",
        "//tensorflow/lite/experimental/shlo:bf16",
        "//tensorflow/lite/experimental/shlo:f16",
        "//tensorflow/lite/experimental/shlo:quantize",
        "//tensorflow/lite/experimental/shlo:quantized_tensor_element_type",
        "//tensorflow/lite/experimental/shlo:shape",
        "//tensorflow/lite/experimental/shlo:status_matcher",
        "//tensorflow/lite/experimental/shlo:tensor",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "floor",
    srcs = ["floor.cc"],
    hdrs = ["floor.h"],
    deps = [
        ":unary_elementwise",
        ":util",
        "//tensorflow/lite/experimental/shlo:bf16",
        "//tensorflow/lite/experimental/shlo:dispatch",
        "//tensorflow/lite/experimental/shlo:f16",
        "//tensorflow/lite/experimental/shlo:tensor",
        "@com_google_absl//absl/status",
    ],
)

cc_test(
    name = "floor_test",
    srcs = ["floor_test.cc"],
    linkopts = shlo_ref_linkopts(),
    deps = [
        ":floor",
        ":test_util",
        ":unary_elementwise_test_util",
        "//tensorflow/lite/experimental/shlo:bf16",
        "//tensorflow/lite/experimental/shlo:f16",
        "//tensorflow/lite/experimental/shlo:quantize",
        "//tensorflow/lite/experimental/shlo:quantized_tensor_element_type",
        "//tensorflow/lite/experimental/shlo:shape",
        "//tensorflow/lite/experimental/shlo:status_matcher",
        "//tensorflow/lite/experimental/shlo:tensor",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "log",
    srcs = ["log.cc"],
    hdrs = ["log.h"],
    deps = [
        ":unary_elementwise",
        ":util",
        "//tensorflow/lite/experimental/shlo:bf16",
        "//tensorflow/lite/experimental/shlo:dispatch",
        "//tensorflow/lite/experimental/shlo:f16",
        "//tensorflow/lite/experimental/shlo:tensor",
        "@com_google_absl//absl/status",
    ],
)

cc_test(
    name = "log_test",
    srcs = ["log_test.cc"],
    linkopts = shlo_ref_linkopts(),
    deps = [
        ":log",
        ":test_util",
        ":unary_elementwise_test_util",
        "//tensorflow/lite/experimental/shlo:bf16",
        "//tensorflow/lite/experimental/shlo:f16",
        "//tensorflow/lite/experimental/shlo:quantize",
        "//tensorflow/lite/experimental/shlo:quantized_tensor_element_type",
        "//tensorflow/lite/experimental/shlo:shape",
        "//tensorflow/lite/experimental/shlo:status_matcher",
        "//tensorflow/lite/experimental/shlo:tensor",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "exponential_minus_one",
    srcs = ["exponential_minus_one.cc"],
    hdrs = ["exponential_minus_one.h"],
    deps = [
        ":unary_elementwise",
        ":util",
        "//tensorflow/lite/experimental/shlo:bf16",
        "//tensorflow/lite/experimental/shlo:dispatch",
        "//tensorflow/lite/experimental/shlo:f16",
        "//tensorflow/lite/experimental/shlo:tensor",
        "@com_google_absl//absl/status",
    ],
)

cc_test(
    name = "exponential_minus_one_test",
    srcs = ["exponential_minus_one_test.cc"],
    linkopts = shlo_ref_linkopts(),
    deps = [
        ":exponential_minus_one",
        ":test_util",
        ":unary_elementwise_test_util",
        "//tensorflow/lite/experimental/shlo:bf16",
        "//tensorflow/lite/experimental/shlo:f16",
        "//tensorflow/lite/experimental/shlo:quantize",
        "//tensorflow/lite/experimental/shlo:quantized_tensor_element_type",
        "//tensorflow/lite/experimental/shlo:shape",
        "//tensorflow/lite/experimental/shlo:status_matcher",
        "//tensorflow/lite/experimental/shlo:tensor",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "log_plus_one",
    srcs = ["log_plus_one.cc"],
    hdrs = ["log_plus_one.h"],
    deps = [
        ":unary_elementwise",
        ":util",
        "//tensorflow/lite/experimental/shlo:bf16",
        "//tensorflow/lite/experimental/shlo:dispatch",
        "//tensorflow/lite/experimental/shlo:f16",
        "//tensorflow/lite/experimental/shlo:tensor",
        "@com_google_absl//absl/status",
    ],
)

cc_test(
    name = "log_plus_one_test",
    srcs = ["log_plus_one_test.cc"],
    linkopts = shlo_ref_linkopts(),
    deps = [
        ":log_plus_one",
        ":test_util",
        ":unary_elementwise_test_util",
        "//tensorflow/lite/experimental/shlo:bf16",
        "//tensorflow/lite/experimental/shlo:f16",
        "//tensorflow/lite/experimental/shlo:quantize",
        "//tensorflow/lite/experimental/shlo:quantized_tensor_element_type",
        "//tensorflow/lite/experimental/shlo:shape",
        "//tensorflow/lite/experimental/shlo:status_matcher",
        "//tensorflow/lite/experimental/shlo:tensor",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "logistic",
    srcs = ["logistic.cc"],
    hdrs = ["logistic.h"],
    deps = [
        ":unary_elementwise",
        ":util",
        "//tensorflow/lite/experimental/shlo:bf16",
        "//tensorflow/lite/experimental/shlo:dispatch",
        "//tensorflow/lite/experimental/shlo:f16",
        "//tensorflow/lite/experimental/shlo:tensor",
        "@com_google_absl//absl/status",
    ],
)

cc_test(
    name = "logistic_test",
    srcs = ["logistic_test.cc"],
    linkopts = shlo_ref_linkopts(),
    deps = [
        ":logistic",
        ":test_util",
        ":unary_elementwise_test_util",
        "//tensorflow/lite/experimental/shlo:bf16",
        "//tensorflow/lite/experimental/shlo:f16",
        "//tensorflow/lite/experimental/shlo:quantize",
        "//tensorflow/lite/experimental/shlo:quantized_tensor_element_type",
        "//tensorflow/lite/experimental/shlo:shape",
        "//tensorflow/lite/experimental/shlo:status_matcher",
        "//tensorflow/lite/experimental/shlo:tensor",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "sign",
    srcs = ["sign.cc"],
    hdrs = ["sign.h"],
    deps = [
        ":unary_elementwise",
        ":util",
        "//tensorflow/lite/experimental/shlo:bf16",
        "//tensorflow/lite/experimental/shlo:dispatch",
        "//tensorflow/lite/experimental/shlo:f16",
        "//tensorflow/lite/experimental/shlo:tensor",
        "@com_google_absl//absl/status",
    ],
)

cc_test(
    name = "sign_test",
    srcs = ["sign_test.cc"],
    linkopts = shlo_ref_linkopts(),
    deps = [
        ":sign",
        ":test_util",
        ":unary_elementwise_test_util",
        "//tensorflow/lite/experimental/shlo:bf16",
        "//tensorflow/lite/experimental/shlo:f16",
        "//tensorflow/lite/experimental/shlo:quantize",
        "//tensorflow/lite/experimental/shlo:quantized_tensor_element_type",
        "//tensorflow/lite/experimental/shlo:shape",
        "//tensorflow/lite/experimental/shlo:status_matcher",
        "//tensorflow/lite/experimental/shlo:tensor",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "negate",
    srcs = ["negate.cc"],
    hdrs = ["negate.h"],
    deps = [
        ":unary_elementwise",
        ":util",
        "//tensorflow/lite/experimental/shlo:dispatch",
        "//tensorflow/lite/experimental/shlo:tensor",
        "@com_google_absl//absl/status",
    ],
)

cc_test(
    name = "negate_test",
    srcs = ["negate_test.cc"],
    linkopts = shlo_ref_linkopts(),
    deps = [
        ":negate",
        ":test_util",
        ":unary_elementwise_test_util",
        "//tensorflow/lite/experimental/shlo:quantize",
        "//tensorflow/lite/experimental/shlo:quantized_tensor_element_type",
        "//tensorflow/lite/experimental/shlo:shape",
        "//tensorflow/lite/experimental/shlo:status_matcher",
        "//tensorflow/lite/experimental/shlo:tensor",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "count_leading_zeros",
    srcs = ["count_leading_zeros.cc"],
    hdrs = ["count_leading_zeros.h"],
    deps = [
        ":unary_elementwise",
        ":util",
        "//tensorflow/lite/experimental/shlo:dispatch",
        "//tensorflow/lite/experimental/shlo:i4",
        "//tensorflow/lite/experimental/shlo:tensor",
        "@com_google_absl//absl/numeric:bits",
        "@com_google_absl//absl/status",
    ],
)

cc_test(
    name = "count_leading_zeros_test",
    srcs = ["count_leading_zeros_test.cc"],
    linkopts = shlo_ref_linkopts(),
    deps = [
        ":count_leading_zeros",
        ":test_util",
        ":unary_elementwise_test_util",
        "//tensorflow/lite/experimental/shlo:data_type",
        "//tensorflow/lite/experimental/shlo:i4",
        "//tensorflow/lite/experimental/shlo:shape",
        "//tensorflow/lite/experimental/shlo:status_matcher",
        "//tensorflow/lite/experimental/shlo:tensor",
        "@com_google_absl//absl/numeric:bits",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "not",
    srcs = ["not.cc"],
    hdrs = ["not.h"],
    deps = [
        ":unary_elementwise",
        ":util",
        "//tensorflow/lite/experimental/shlo:dispatch",
        "//tensorflow/lite/experimental/shlo:tensor",
        "@com_google_absl//absl/status",
    ],
)

cc_test(
    name = "not_test",
    srcs = ["not_test.cc"],
    deps = [
        ":not",
        ":test_util",
        ":unary_elementwise_test_util",
        "//tensorflow/lite/experimental/shlo:data_type",
        "//tensorflow/lite/experimental/shlo:shape",
        "//tensorflow/lite/experimental/shlo:status_matcher",
        "//tensorflow/lite/experimental/shlo:tensor",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "popcnt",
    srcs = ["popcnt.cc"],
    hdrs = ["popcnt.h"],
    deps = [
        ":unary_elementwise",
        ":util",
        "//tensorflow/lite/experimental/shlo:dispatch",
        "//tensorflow/lite/experimental/shlo:i4",
        "//tensorflow/lite/experimental/shlo:tensor",
        "@com_google_absl//absl/numeric:bits",
        "@com_google_absl//absl/status",
    ],
)

cc_test(
    name = "popcnt_test",
    srcs = ["popcnt_test.cc"],
    deps = [
        ":popcnt",
        ":test_util",
        ":unary_elementwise_test_util",
        "//tensorflow/lite/experimental/shlo:data_type",
        "//tensorflow/lite/experimental/shlo:i4",
        "//tensorflow/lite/experimental/shlo:shape",
        "//tensorflow/lite/experimental/shlo:status_matcher",
        "//tensorflow/lite/experimental/shlo:tensor",
        "@com_google_absl//absl/numeric:bits",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "binary_elementwise",
    hdrs = ["binary_elementwise.h"],
    deps = [
        "//tensorflow/lite/experimental/shlo:data_type",
        "//tensorflow/lite/experimental/shlo:quantize",
        "//tensorflow/lite/experimental/shlo:quantized_tensor_element_type",
        "//tensorflow/lite/experimental/shlo:shape",
        "//tensorflow/lite/experimental/shlo:tensor",
    ],
)

cc_test(
    name = "binary_elementwise_test",
    srcs = ["binary_elementwise_test.cc"],
    linkopts = shlo_ref_linkopts(),
    deps = [
        ":binary_elementwise",
        ":test_util",
        "//tensorflow/lite/experimental/shlo:quantized_tensor_element_type",
        "//tensorflow/lite/experimental/shlo:shape",
        "//tensorflow/lite/experimental/shlo:tensor",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "binary_elementwise_test_util",
    testonly = True,
    hdrs = ["binary_elementwise_test_util.h"],
    deps = [
        ":test_util",
        "//tensorflow/lite/experimental/shlo:data_type",
        "//tensorflow/lite/experimental/shlo:shape",
        "//tensorflow/lite/experimental/shlo:status_matcher",
        "//tensorflow/lite/experimental/shlo:tensor",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "multiply",
    srcs = ["multiply.cc"],
    hdrs = ["multiply.h"],
    deps = [
        ":binary_elementwise",
        ":util",
        "//tensorflow/lite/experimental/shlo:data_type",
        "//tensorflow/lite/experimental/shlo:dispatch",
        "//tensorflow/lite/experimental/shlo:tensor",
        "@com_google_absl//absl/status",
    ],
)

cc_test(
    name = "multiply_test",
    srcs = ["multiply_test.cc"],
    deps = [
        ":binary_elementwise_test_util",
        ":multiply",
        ":test_util",
        "//tensorflow/lite/experimental/shlo:data_type",
        "//tensorflow/lite/experimental/shlo:quantize",
        "//tensorflow/lite/experimental/shlo:quantized_tensor_element_type",
        "//tensorflow/lite/experimental/shlo:shape",
        "//tensorflow/lite/experimental/shlo:status_matcher",
        "//tensorflow/lite/experimental/shlo:tensor",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "divide",
    srcs = ["divide.cc"],
    hdrs = ["divide.h"],
    deps = [
        ":binary_elementwise",
        ":util",
        "//tensorflow/lite/experimental/shlo:dispatch",
        "//tensorflow/lite/experimental/shlo:tensor",
        "@com_google_absl//absl/status",
    ],
)

cc_test(
    name = "divide_test",
    srcs = ["divide_test.cc"],
    deps = [
        ":binary_elementwise_test_util",
        ":divide",
        ":test_util",
        "//tensorflow/lite/experimental/shlo:quantize",
        "//tensorflow/lite/experimental/shlo:quantized_tensor_element_type",
        "//tensorflow/lite/experimental/shlo:shape",
        "//tensorflow/lite/experimental/shlo:status_matcher",
        "//tensorflow/lite/experimental/shlo:tensor",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "subtract",
    srcs = ["subtract.cc"],
    hdrs = ["subtract.h"],
    deps = [
        ":binary_elementwise",
        ":util",
        "//tensorflow/lite/experimental/shlo:dispatch",
        "//tensorflow/lite/experimental/shlo:tensor",
        "@com_google_absl//absl/status",
    ],
)

cc_test(
    name = "subtract_test",
    srcs = ["subtract_test.cc"],
    deps = [
        ":binary_elementwise_test_util",
        ":subtract",
        ":test_util",
        "//tensorflow/lite/experimental/shlo:quantize",
        "//tensorflow/lite/experimental/shlo:quantized_tensor_element_type",
        "//tensorflow/lite/experimental/shlo:shape",
        "//tensorflow/lite/experimental/shlo:status_matcher",
        "//tensorflow/lite/experimental/shlo:tensor",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "and",
    srcs = ["and.cc"],
    hdrs = ["and.h"],
    deps = [
        ":binary_elementwise",
        ":util",
        "//tensorflow/lite/experimental/shlo:data_type",
        "//tensorflow/lite/experimental/shlo:dispatch",
        "//tensorflow/lite/experimental/shlo:tensor",
        "@com_google_absl//absl/status",
    ],
)

cc_test(
    name = "and_test",
    srcs = ["and_test.cc"],
    deps = [
        ":and",
        ":binary_elementwise_test_util",
        ":test_util",
        "//tensorflow/lite/experimental/shlo:data_type",
        "//tensorflow/lite/experimental/shlo:shape",
        "//tensorflow/lite/experimental/shlo:status_matcher",
        "//tensorflow/lite/experimental/shlo:tensor",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "or",
    srcs = ["or.cc"],
    hdrs = ["or.h"],
    deps = [
        ":binary_elementwise",
        ":util",
        "//tensorflow/lite/experimental/shlo:data_type",
        "//tensorflow/lite/experimental/shlo:dispatch",
        "//tensorflow/lite/experimental/shlo:tensor",
        "@com_google_absl//absl/status",
    ],
)

cc_test(
    name = "or_test",
    srcs = ["or_test.cc"],
    deps = [
        ":binary_elementwise_test_util",
        ":or",
        ":test_util",
        "//tensorflow/lite/experimental/shlo:data_type",
        "//tensorflow/lite/experimental/shlo:shape",
        "//tensorflow/lite/experimental/shlo:status_matcher",
        "//tensorflow/lite/experimental/shlo:tensor",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "xor",
    srcs = ["xor.cc"],
    hdrs = ["xor.h"],
    deps = [
        ":binary_elementwise",
        ":util",
        "//tensorflow/lite/experimental/shlo:data_type",
        "//tensorflow/lite/experimental/shlo:dispatch",
        "//tensorflow/lite/experimental/shlo:tensor",
        "@com_google_absl//absl/status",
    ],
)

cc_test(
    name = "xor_test",
    srcs = ["xor_test.cc"],
    deps = [
        ":binary_elementwise_test_util",
        ":test_util",
        ":xor",
        "//tensorflow/lite/experimental/shlo:data_type",
        "//tensorflow/lite/experimental/shlo:shape",
        "//tensorflow/lite/experimental/shlo:status_matcher",
        "//tensorflow/lite/experimental/shlo:tensor",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "maximum",
    srcs = ["maximum.cc"],
    hdrs = ["maximum.h"],
    deps = [
        ":binary_elementwise",
        ":util",
        "//tensorflow/lite/experimental/shlo:dispatch",
        "//tensorflow/lite/experimental/shlo:tensor",
        "@com_google_absl//absl/status",
    ],
)

cc_test(
    name = "maximum_test",
    srcs = ["maximum_test.cc"],
    deps = [
        ":binary_elementwise_test_util",
        ":maximum",
        ":test_util",
        "//tensorflow/lite/experimental/shlo:quantize",
        "//tensorflow/lite/experimental/shlo:quantized_tensor_element_type",
        "//tensorflow/lite/experimental/shlo:shape",
        "//tensorflow/lite/experimental/shlo:status_matcher",
        "//tensorflow/lite/experimental/shlo:tensor",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "minimum",
    srcs = ["minimum.cc"],
    hdrs = ["minimum.h"],
    deps = [
        ":binary_elementwise",
        ":util",
        "//tensorflow/lite/experimental/shlo:dispatch",
        "//tensorflow/lite/experimental/shlo:tensor",
        "@com_google_absl//absl/status",
    ],
)

cc_test(
    name = "minimum_test",
    srcs = ["minimum_test.cc"],
    deps = [
        ":binary_elementwise_test_util",
        ":minimum",
        ":test_util",
        "//tensorflow/lite/experimental/shlo:quantize",
        "//tensorflow/lite/experimental/shlo:quantized_tensor_element_type",
        "//tensorflow/lite/experimental/shlo:shape",
        "//tensorflow/lite/experimental/shlo:status_matcher",
        "//tensorflow/lite/experimental/shlo:tensor",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "compare",
    srcs = ["compare.cc"],
    hdrs = ["compare.h"],
    deps = [
        ":binary_elementwise",
        ":util",
        "//tensorflow/lite/experimental/shlo:data_type",
        "//tensorflow/lite/experimental/shlo:dispatch",
        "//tensorflow/lite/experimental/shlo:shape",
        "//tensorflow/lite/experimental/shlo:tensor",
        "@com_google_absl//absl/status",
    ],
)

cc_test(
    name = "compare_test",
    srcs = ["compare_test.cc"],
    deps = [
        ":binary_elementwise_test_util",
        ":compare",
        ":test_util",
        "//tensorflow/lite/experimental/shlo:data_type",
        "//tensorflow/lite/experimental/shlo:quantize",
        "//tensorflow/lite/experimental/shlo:quantized_tensor_element_type",
        "//tensorflow/lite/experimental/shlo:shape",
        "//tensorflow/lite/experimental/shlo:status_matcher",
        "//tensorflow/lite/experimental/shlo:tensor",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/random:distributions",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)

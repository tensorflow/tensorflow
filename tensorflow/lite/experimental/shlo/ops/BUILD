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
        ":benchmark_util",
        ":is_finite",
        "//tensorflow/lite/experimental/shlo:bf16",
        "//tensorflow/lite/experimental/shlo:data_type",
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
        "//tensorflow/lite/experimental/shlo:shape",
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
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "test_util",
    testonly = True,
    hdrs = ["test_util.h"],
    deps = [
        "//tensorflow/lite/experimental/shlo:data_type",
        "//tensorflow/lite/experimental/shlo:shape",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:inlined_vector",
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
        "//tensorflow/lite/experimental/shlo:data_type",
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
        "//tensorflow/lite/experimental/shlo:bf16",
        "//tensorflow/lite/experimental/shlo:f16",
        "//tensorflow/lite/experimental/shlo:quantize",
        "//tensorflow/lite/experimental/shlo:quantized_tensor_element_type",
        "//tensorflow/lite/experimental/shlo:shape",
        "//tensorflow/lite/experimental/shlo:status_matcher",
        "//tensorflow/lite/experimental/shlo:tensor",
        "@com_google_absl//absl/status",
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
        "//tensorflow/lite/experimental/shlo:data_type",
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
        "//tensorflow/lite/experimental/shlo:bf16",
        "//tensorflow/lite/experimental/shlo:f16",
        "//tensorflow/lite/experimental/shlo:quantize",
        "//tensorflow/lite/experimental/shlo:quantized_tensor_element_type",
        "//tensorflow/lite/experimental/shlo:shape",
        "//tensorflow/lite/experimental/shlo:status_matcher",
        "//tensorflow/lite/experimental/shlo:tensor",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "cosine",
    srcs = ["cosine.cc"],
    hdrs = ["cosine.h"],
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
    name = "cosine_test",
    srcs = ["cosine_test.cc"],
    deps = [
        ":cosine",
        ":test_util",
        "//tensorflow/lite/experimental/shlo:bf16",
        "//tensorflow/lite/experimental/shlo:f16",
        "//tensorflow/lite/experimental/shlo:quantize",
        "//tensorflow/lite/experimental/shlo:quantized_tensor_element_type",
        "//tensorflow/lite/experimental/shlo:shape",
        "//tensorflow/lite/experimental/shlo:tensor",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)

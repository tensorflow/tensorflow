load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

# StableHLO Reference Library
load("build_def.bzl", "shlo_ref_linkopts")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:LICENSE"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "shlo",
    deps = [
        ":tensor",
    ],
)

cc_library(
    name = "tensor",
    srcs = ["tensor.cc"],
    hdrs = ["tensor.h"],
    deps = [
        ":data_type",
        ":overload",
        ":quantized_tensor_element_type",
        ":shape",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "i4",
    hdrs = ["i4.h"],
    deps = [],
)

cc_test(
    name = "i4_test",
    srcs = ["i4_test.cc"],
    deps = [
        ":i4",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "tensor_test",
    srcs = ["tensor_test.cc"],
    deps = [
        ":data_type",
        ":tensor",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "shape",
    srcs = ["shape.cc"],
    hdrs = ["shape.h"],
    deps = [
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "shape_test",
    srcs = ["shape_test.cc"],
    linkopts = shlo_ref_linkopts(),
    deps = [
        ":shape",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "quantized_tensor_element_type",
    srcs = ["quantized_tensor_element_type.cc"],
    hdrs = ["quantized_tensor_element_type.h"],
    deps = [
        ":data_type",
        ":shape",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "quantized_tensor_element_type_test",
    srcs = ["quantized_tensor_element_type_test.cc"],
    linkopts = shlo_ref_linkopts(),
    deps = [
        ":data_type",
        ":quantized_tensor_element_type",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "bf16",
    hdrs = ["bf16.h"],
    deps = [
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/log:absl_check",
    ],
)

cc_test(
    name = "bf16_test",
    srcs = ["bf16_test.cc"],
    linkopts = shlo_ref_linkopts(),
    deps = [
        ":bf16",
        "@com_google_absl//absl/base",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "f16",
    hdrs = ["f16.h"],
    deps = ["@FP16"],
)

cc_library(
    name = "f16_emulated",
    hdrs = ["f16.h"],
    copts = ["-DSHLO_REF_EMULATE_F16=1"],
    deps = ["@FP16"],
)

cc_test(
    name = "f16_test",
    srcs = ["f16_test.cc"],
    linkopts = shlo_ref_linkopts(),
    deps = [
        ":f16",
        "@com_google_absl//absl/base",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "f16_emulated_test",
    srcs = ["f16_test.cc"],
    linkopts = shlo_ref_linkopts(),
    deps = [
        ":f16_emulated",
        "@com_google_absl//absl/base",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "data_type",
    hdrs = ["data_type.h"],
    deps = [
        ":bf16",
        ":f16",
        ":i4",
    ],
)

cc_library(
    name = "dispatch",
    hdrs = ["dispatch.h"],
    visibility = ["//tensorflow/lite/experimental/shlo:__subpackages__"],
)

cc_test(
    name = "dispatch_test",
    srcs = ["dispatch_test.cc"],
    linkopts = shlo_ref_linkopts(),
    deps = [
        ":dispatch",
        ":status_matcher",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "overload",
    hdrs = ["overload.h"],
    visibility = ["//tensorflow/lite/experimental/shlo:__subpackages__"],
)

cc_test(
    name = "overload_test",
    srcs = ["overload_test.cc"],
    deps = [
        ":overload",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "quantize",
    hdrs = ["quantize.h"],
    deps = [":data_type"],
)

cc_test(
    name = "quantize_test",
    srcs = ["quantize_test.cc"],
    linkopts = shlo_ref_linkopts(),
    deps = [
        ":data_type",
        ":quantize",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "status_matcher",
    testonly = True,
    hdrs = ["status_matcher.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "tensor_with_data",
    testonly = True,
    hdrs = ["tensor_with_data.h"],
    deps = [
        ":data_type",
        ":quantize",
        ":quantized_tensor_element_type",
        ":shape",
        ":tensor",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "tensor_matcher",
    testonly = True,
    hdrs = ["tensor_matcher.h"],
    deps = [
        ":data_type",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "tensor_matcher_test",
    srcs = ["tensor_matcher_test.cc"],
    linkopts = shlo_ref_linkopts(),
    deps = [
        ":data_type",
        ":shape",
        ":tensor_matcher",
        ":tensor_with_data",
        "@com_google_googletest//:gtest_main",
    ],
)

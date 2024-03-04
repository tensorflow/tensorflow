# StableHLO Reference Library

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
        ":quantized_tensor_element_type",
        ":shape",
        "@com_google_absl//absl/types:span",
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
    deps = [
        ":shape",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "quantized_tensor_element_type",
    hdrs = ["quantized_tensor_element_type.h"],
    deps = [
        ":data_type",
        ":shape",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "quantized_tensor_element_type_test",
    srcs = ["quantized_tensor_element_type_test.cc"],
    deps = [
        ":data_type",
        ":f16",
        ":quantized_tensor_element_type",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "bf16",
    hdrs = ["bf16.h"],
    deps = [
        ":has_keyword",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/log:absl_check",
    ],
)

cc_test(
    name = "bf16_test",
    srcs = ["bf16_test.cc"],
    deps = [
        ":bf16",
        "@com_google_absl//absl/base",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "f16",
    hdrs = ["f16.h"],
    deps = [":has_keyword"],
)

cc_library(
    name = "has_keyword",
    hdrs = ["has_keyword.h"],
)

cc_library(
    name = "data_type",
    hdrs = ["data_type.h"],
    deps = [
        ":bf16",
        ":f16",
    ],
)

cc_library(
    name = "dispatch",
    hdrs = ["dispatch.h"],
    visibility = ["//tensorflow/lite/experimental/shlo:__subpackages__"],
    deps = [":data_type"],
)

cc_test(
    name = "dispatch_test",
    srcs = ["dispatch_test.cc"],
    deps = [
        ":dispatch",
        "@com_google_absl//absl/status",
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
    deps = [
        ":data_type",
        ":quantize",
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
    deps = [
        ":data_type",
        ":shape",
        ":tensor_matcher",
        ":tensor_with_data",
        "@com_google_googletest//:gtest_main",
    ],
)

# StableHLO Reference Library

package(
    default_applicable_licenses = ["//tensorflow:LICENSE"],
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
        ":quantized_tensor_element_type",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "bf16",
    hdrs = ["bf16.h"],
    deps = [":has_keyword"],
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
    visibility = ["//visibility:private"],
    deps = [":data_type"],
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

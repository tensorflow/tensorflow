package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "convert",
    srcs = ["convert.cc"],
    hdrs = ["convert.h"],
    deps = [
        ":shape",
        ":status",
        ":tensor",
        ":types",
        ":util",
        "@FP16",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "custom_parsers",
    srcs = ["custom_parsers.cc"],
    hdrs = ["custom_parsers.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:any",
        "@flatbuffers",
    ],
)

cc_library(
    name = "access_type",
    hdrs = ["access_type.h"],
)

cc_library(
    name = "gpu_info",
    srcs = ["gpu_info.cc"],
    hdrs = ["gpu_info.h"],
    deps = [
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "data_type",
    srcs = ["data_type.cc"],
    hdrs = ["data_type.h"],
)

cc_library(
    name = "memory_management",
    srcs = [
        "memory_management.cc",
        "memory_management/greedy_by_breadth_assignment.cc",
        "memory_management/greedy_by_size_assignment.cc",
        "memory_management/internal.cc",
        "memory_management/min_cost_flow_assignment.cc",
        "memory_management/types.cc",
    ],
    hdrs = [
        "memory_management.h",
        "memory_management/equality_assignment.h",
        "memory_management/greedy_by_breadth_assignment.h",
        "memory_management/greedy_by_size_assignment.h",
        "memory_management/greedy_in_order_assignment.h",
        "memory_management/internal.h",
        "memory_management/min_cost_flow_assignment.h",
        "memory_management/naive_assignment.h",
        "memory_management/types.h",
    ],
    deps = [
        ":shape",
        ":status",
        ":types",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "model",
    hdrs = ["model.h"],
    deps = [
        ":data_type",
        ":shape",
        ":status",
        ":tensor",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/types:any",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_test(
    name = "model_test",
    srcs = ["model_test.cc"],
    deps = [
        ":model",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "model_builder",
    srcs = ["model_builder.cc"],
    hdrs = ["model_builder.h"],
    deps = [
        ":custom_parsers",
        ":data_type",
        ":model",
        ":operations",
        ":shape",
        ":status",
        ":tensor",
        "//tensorflow/lite:context",
        "//tensorflow/lite:kernel_api",
        "//tensorflow/lite:util",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/delegates:utils",
        "//tensorflow/lite/delegates/gpu/common/transformations:general_transformations",
        "//tensorflow/lite/kernels:kernel_util",
        "//tensorflow/lite/schema:schema_fbs",
        "@FP16",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_test(
    name = "model_builder_test",
    srcs = ["model_builder_test.cc"],
    deps = [
        ":model_builder",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:kernel_api",
        "//tensorflow/lite/c:common",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "model_transformer",
    srcs = ["model_transformer.cc"],
    hdrs = ["model_transformer.h"],
    deps = [
        ":model",
        "@com_google_absl//absl/strings",
    ],
)

# TODO(impjdi): Add unit test for model_transformer.

cc_library(
    name = "operations",
    srcs = ["operations.cc"],
    hdrs = ["operations.h"],
    deps = [
        ":data_type",
        ":model",
        ":shape",
        ":status",
        "@com_google_absl//absl/types:variant",
    ],
)

# TODO(impjdi): Add unit test for operations.

cc_library(
    name = "shape",
    srcs = ["shape.cc"],
    hdrs = ["shape.h"],
    deps = [
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/strings",
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
    name = "status",
    hdrs = ["status.h"],
)

cc_library(
    name = "tensor",
    hdrs = ["tensor.h"],
    deps = [
        ":data_type",
        ":shape",
    ],
)

cc_library(
    name = "types",
    hdrs = ["types.h"],
    deps = [
        "@FP16",
    ],
)

cc_library(
    name = "util",
    hdrs = ["util.h"],
    deps = [
        ":types",
    ],
)

cc_test(
    name = "memory_management_test",
    srcs = ["memory_management_test.cc"],
    deps = [
        ":memory_management",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "memory_management_internal_test",
    srcs = ["memory_management/internal_test.cc"],
    deps = [
        ":memory_management",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "memory_management_types_test",
    srcs = ["memory_management/types_test.cc"],
    deps = [
        ":memory_management",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "util_test",
    srcs = ["util_test.cc"],
    deps = [
        ":util",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "workgroup_selection",
    srcs = ["workgroup_selection.cc"],
    hdrs = ["workgroup_selection.h"],
    deps = [
        ":status",
        ":types",
        ":util",
    ],
)

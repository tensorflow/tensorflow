load("//tensorflow/core/platform:build_config.bzl", "tf_platform_alias")
load("@flatbuffers//:build_defs.bzl", "flatbuffer_cc_library")
load("//tensorflow:tensorflow.bzl", "workspace_root")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "convert",
    srcs = ["convert.cc"],
    hdrs = ["convert.h"],
    deps = [
        ":data_type",
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

exports_files(
    [
        "custom_parsers.h",
        "custom_transformations.h",
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
        ":data_type",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "gpu_model",
    srcs = ["gpu_model.cc"],
    hdrs = ["gpu_model.h"],
    deps = [
        ":model",
        ":model_hints",
        ":operations",
        ":status",
        "//tensorflow/lite/delegates/gpu/common:gpu_model_cc_fbs",
        "//tensorflow/lite/delegates/gpu/common/selectors:operation_selector",
        "//tensorflow/lite/delegates/gpu/common/selectors:special_selector",
        "//tensorflow/lite/delegates/gpu/common/selectors:subgraph",
        "//tensorflow/lite/delegates/gpu/common/task:gpu_operation",
        "//tensorflow/lite/delegates/gpu/common/task:serialization_base",
        "//tensorflow/lite/delegates/gpu/common/task:storage_type_util",
        "//tensorflow/lite/delegates/gpu/common/transformations:add_bias",
        "//tensorflow/lite/delegates/gpu/common/transformations:global_pooling_to_reduce_op",
        "//tensorflow/lite/delegates/gpu/common/transformations:merge_padding_with",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

flatbuffer_cc_library(
    name = "gpu_model_cc_fbs",
    srcs = ["gpu_model.fbs"],
    flatc_args = [
        "--scoped-enums",
        "-I " + workspace_root,
    ],
    includes = [
        "//tensorflow/lite/delegates/gpu/common/task:serialization_base_cc_fbs_includes",
    ],
)

cc_library(
    name = "kernel_info",
    hdrs = ["kernel_info.h"],
)

cc_library(
    name = "data_type",
    srcs = ["data_type.cc"],
    hdrs = ["data_type.h"],
    deps = [
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "data_type_test",
    srcs = ["data_type_test.cc"],
    deps = [
        ":data_type",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "flops_util",
    srcs = ["flops_util.cc"],
    hdrs = ["flops_util.h"],
    deps = [
        ":shape",
    ],
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
        ":util",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "model",
    srcs = ["model.cc"],
    hdrs = ["model.h"],
    deps = [
        ":shape",
        ":status",
        ":tensor",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:any",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_test(
    name = "model_test",
    srcs = ["model_test.cc"],
    deps = [
        ":model",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "lstm_parser",
    srcs = ["lstm_parser.cc"],
    hdrs = ["lstm_parser.h"],
    deps = [
        ":data_type",
        ":model",
        ":model_builder_helper",
        ":object_reader",
        ":operations",
        ":shape",
        ":status",
        ":tensor",
        "//tensorflow/lite:string",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels:lstm_shared",
        "//tensorflow/lite/kernels/internal:quantization_util",
        "//tensorflow/lite/kernels/internal:tensor",
        "//tensorflow/lite/kernels/internal:tensor_utils",
        "//tensorflow/lite/kernels/internal:types",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:any",
    ],
)

cc_library(
    name = "model_builder",
    srcs = ["model_builder.cc"],
    hdrs = [
        "model_builder.h",
        "model_builder_internal.h",
    ],
    deps = [
        ":data_type",
        ":lstm_parser",
        ":model",
        ":model_builder_helper",
        ":model_transformer",
        ":object_reader",
        ":operation_parser",
        ":operations",
        ":shape",
        ":status",
        ":tensor",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/delegates:utils",
        "//tensorflow/lite:framework_lib",
        "//tensorflow/lite:kernel_api",
        "//tensorflow/lite:util",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/tools/versioning:gpu_compatibility",
        "//tensorflow/lite/delegates/gpu/common/transformations:model_transformations",
        "//tensorflow/lite/kernels:kernel_util",
        "//tensorflow/lite/kernels/internal:reference_base",
        "//tensorflow/lite/kernels/internal:tensor",
    ] + tf_platform_alias("custom_parsers", "//tensorflow/lite/delegates/gpu/common/"),
)

cc_test(
    name = "model_builder_test",
    srcs = ["model_builder_test.cc"],
    deps = [
        ":data_type",
        ":model_builder",
        ":shape",
        ":tensor",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:kernel_api",
        "//tensorflow/lite/c:common",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "model_builder_helper",
    srcs = ["model_builder_helper.cc"],
    hdrs = ["model_builder_helper.h"],
    deps = [
        ":data_type",
        ":model",
        ":operations",
        ":shape",
        ":status",
        ":tensor",
        "//tensorflow/lite:kernel_api",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels:kernel_util",
        "//tensorflow/lite/kernels/internal:reference_base",
        "//tensorflow/lite/kernels/internal:tensor",
        "//tensorflow/lite/kernels/internal:types",
        "@FP16",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "model_hints",
    hdrs = ["model_hints.h"],
)

cc_library(
    name = "model_transformer",
    srcs = ["model_transformer.cc"],
    hdrs = ["model_transformer.h"],
    deps = [
        ":model",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

# TODO(impjdi): Add unit test for model_transformer.

cc_library(
    name = "object_reader",
    srcs = ["object_reader.cc"],
    hdrs = ["object_reader.h"],
    deps = [
        ":model",
        ":model_builder_helper",
        ":status",
        ":tensor",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/delegates:utils",
        "//tensorflow/lite/kernels:kernel_util",
        "//tensorflow/lite/kernels/internal/utils:sparsity_format_converter",
        "@FP16",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "operations",
    srcs = ["operations.cc"],
    hdrs = ["operations.h"],
    deps = [
        ":data_type",
        ":shape",
        ":status",
        ":tensor",
        "@com_google_absl//absl/types:variant",
    ],
)

cc_library(
    name = "precision",
    srcs = ["precision.cc"],
    hdrs = ["precision.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:data_type",
    ],
)

cc_library(
    name = "quantization_util",
    srcs = ["quantization_util.cc"],
    hdrs = ["quantization_util.h"],
    deps = [
        ":status",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels/internal:optimized_base",
        "//tensorflow/lite/kernels/internal:tensor",
        "//tensorflow/lite/kernels/internal:types",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
    ],
)

cc_test(
    name = "quantization_util_test",
    srcs = ["quantization_util_test.cc"],
    deps = [
        ":quantization_util",
        "//tensorflow/lite:util",
        "//tensorflow/lite/c:common",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)

# TODO(impjdi): Add unit test for operations.

cc_library(
    name = "shape",
    srcs = ["shape.cc"],
    hdrs = ["shape.h"],
    deps = ["@com_google_absl//absl/strings"],
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
    deps = [
        "//tensorflow/lite/delegates/gpu/common/google:status_macros",
        "@com_google_absl//absl/status",
    ],
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
        ":shape",
        ":types",
        "@com_google_absl//absl/status",
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

cc_library(
    name = "operation_parser",
    srcs = ["operation_parser.cc"],
    hdrs = ["operation_parser.h"],
    deps = [
        ":model",
        ":object_reader",
        ":operations",
        ":shape",
        ":status",
        "//tensorflow/lite/c:common",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "unimplemented_operation_parser",
    hdrs = ["unimplemented_operation_parser.h"],
    deps = [
        ":operation_parser",
        ":status",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:any",
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
    name = "winograd_util",
    srcs = ["winograd_util.cc"],
    hdrs = ["winograd_util.h"],
    deps = [
        ":data_type",
        ":operations",
        ":shape",
        ":tensor",
    ],
)

cc_test(
    name = "winograd_util_test",
    srcs = ["winograd_util_test.cc"],
    deps = [
        ":winograd_util",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "workgroup_selection",
    srcs = ["workgroup_selection.cc"],
    hdrs = ["workgroup_selection.h"],
    deps = [":util"],
)

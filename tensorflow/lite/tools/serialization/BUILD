load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    features = ["-parse_headers"],
    licenses = ["notice"],
)

cc_binary(
    name = "option_writer_generator",
    srcs = ["option_writer_generator.cc"],
    deps = [
        "//tensorflow/lite/schema:schema_fbs_with_reflection",
        "@flatbuffers",
    ],
)

cc_library(
    name = "writer_lib_enum",
    hdrs = ["enum_mapping.h"],
    deps = [
        "//tensorflow/compiler/mlir/lite/schema:schema_fbs_with_mutable",
        "//tensorflow/lite:builtin_op_data",
    ],
)

cc_library(
    name = "writer_lib",
    srcs = [
        "writer_lib.cc",
    ],
    hdrs = [
        "writer_lib.h",
    ],
    data = [
        ":option_writer_gen",
    ],
    textual_hdrs = ["option_writer_generated.h"],
    deps = [
        "//tensorflow/compiler/mlir/lite/schema:schema_conversion_utils",
        "//tensorflow/compiler/mlir/lite/schema:schema_fbs_with_mutable",
        "//tensorflow/compiler/mlir/lite/tools/versioning",
        "//tensorflow/lite:builtin_op_data",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/core:framework_stable",
        "//tensorflow/lite/core:model_builder",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/schema:schema_conversion_utils",
        "//tensorflow/lite/tools/serialization:writer_lib_enum",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@flatbuffers//:runtime_cc",
    ],
)

cc_binary(
    name = "writer",
    srcs = ["writer.cc"],
    deps = [
        ":writer_lib",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/kernels:builtin_ops",
    ],
)

cc_binary(
    name = "writer_test",
    srcs = ["writer_test.cc"],
    deps = [
        ":writer_lib",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/kernels:builtin_ops",
    ],
)

cc_test(
    name = "writer_lib_test",
    size = "small",
    srcs = ["writer_lib_test.cc"],
    deps = [
        ":writer_lib",
        "//tensorflow/compiler/mlir/lite/schema:schema_fbs_with_mutable",
        "//tensorflow/lite:builtin_ops",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:util",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:c_api_types",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
        "//tensorflow/lite/kernels:subgraph_test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_absl//absl/log:check",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:logging",
    ],
)

genrule(
    name = "option_writer_gen",
    outs = ["option_writer_generated.h"],
    cmd = "$(location :option_writer_generator) $(@)",
    tools = [":option_writer_generator"],
)

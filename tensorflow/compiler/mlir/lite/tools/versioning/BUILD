load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
)
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "versioning",
    srcs = [
        "op_version.cc",
        "runtime_version.cc",
    ],
    hdrs = [
        "op_version.h",
        "runtime_version.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":op_signature",
        "//tensorflow/compiler/mlir/lite/core/c:tflite_common",
        "//tensorflow/compiler/mlir/lite/kernels/internal:compatibility_macros",
        "//tensorflow/compiler/mlir/lite/schema:schema_fbs",
        "//tensorflow/compiler/mlir/lite/schema:schema_fbs_with_mutable",
        "//tensorflow/compiler/mlir/lite/schema:schema_utils",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@flatbuffers",
    ],
)

tf_cc_test(
    name = "versioning_test",
    srcs = [
        "op_version_test.cc",
    ],
    deps = [
        ":op_signature",
        ":versioning",
        "//tensorflow/compiler/mlir/lite/core/c:tflite_common",
        "//tensorflow/compiler/mlir/lite/schema:schema_fbs",
        "//tensorflow/compiler/mlir/lite/schema:schema_fbs_with_mutable",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "op_signature",
    srcs = [
        "op_signature.cc",
    ],
    hdrs = [
        "op_signature.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tensorflow/compiler/mlir/lite/core/api:flatbuffer_conversions",
        "//tensorflow/compiler/mlir/lite/core/c:tflite_common",
        "//tensorflow/compiler/mlir/lite/schema:schema_fbs",
        "//tensorflow/compiler/mlir/lite/schema:schema_utils",
        "@flatbuffers//:runtime_cc",
    ],
)

tf_cc_test(
    name = "op_signature_test",
    srcs = [
        "op_signature_test.cc",
    ],
    data = [
        "//tensorflow/compiler/mlir/lite:testdata/add.bin",
        "//tensorflow/compiler/mlir/lite:testdata/multi_signatures.bin",
    ],
    deps = [
        ":op_signature",
        "//tensorflow/compiler/mlir/lite/core:absl_error_model_builder",
        "//tensorflow/compiler/mlir/lite/core/c:tflite_common",
        "//tensorflow/compiler/mlir/lite/schema:schema_fbs",
        "//tensorflow/core/platform:resource_loader",
        "@com_google_googletest//:gtest_main",
    ],
)

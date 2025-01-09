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
        "//tensorflow/compiler/mlir/lite/schema:schema_fbs",
        "//tensorflow/compiler/mlir/lite/schema:schema_fbs_with_mutable",
        "//tensorflow/core:tflite_portable_logging",
        "//tensorflow/lite:builtin_op_data",
        "//tensorflow/lite:minimal_logging",
        "//tensorflow/lite/core/c:c_api_types",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/kernels/internal:compatibility",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/schema:schema_utils",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@flatbuffers",
    ],
)

tf_cc_test(
    name = "versioning_test",
    srcs = [
        "op_version_test.cc",
        "runtime_version_test.cc",
    ],
    deps = [
        ":versioning",
        "//tensorflow/lite:builtin_op_data",
        "//tensorflow/lite/core/c:c_api_types",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/schema:schema_fbs_with_mutable",
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
        "//tensorflow/compiler/mlir/lite/tools/versioning:op_signature",
        "//tensorflow/lite:stderr_reporter",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/core/c:c_api_types",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/kernels:kernel_util",
        "//tensorflow/lite/kernels/internal:compatibility",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/schema:schema_utils",
    ],
)

tf_cc_test(
    name = "op_signature_test",
    srcs = [
        "op_signature_test.cc",
    ],
    data = [
        "//tensorflow/lite:testdata/add.bin",
        "//tensorflow/lite:testdata/multi_signatures.bin",
    ],
    deps = [
        ":op_signature",
        "//tensorflow/core/platform:resource_loader",
        "//tensorflow/lite:builtin_ops",
        "//tensorflow/lite/core:model_builder",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "gpu_compatibility",
    srcs = [
        "gpu_compatibility.cc",
    ],
    hdrs = [
        "gpu_compatibility.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":op_signature",
        "//tensorflow/lite:builtin_op_data",
        "//tensorflow/lite:builtin_ops",
        "//tensorflow/lite:util",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "gpu_compatibility_test",
    srcs = [
        "gpu_compatibility_test.cc",
    ],
    data = [
        "//tensorflow/lite:testdata/conv3d_huge_im2col.bin",
        "//tensorflow/lite:testdata/conv_huge_im2col.bin",
        "//tensorflow/lite:testdata/multi_add_flex.bin",
    ],
    deps = [
        ":gpu_compatibility",
        ":op_signature",
        "//tensorflow/core/platform:resource_loader",
        "//tensorflow/lite/core:model_builder",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/kernels/internal:types",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)

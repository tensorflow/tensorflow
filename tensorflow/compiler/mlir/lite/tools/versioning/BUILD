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
        "//tensorflow/compiler/mlir/lite/core/c:c_api_types",
        "//tensorflow/compiler/mlir/lite/core/c:tflite_common",
        "//tensorflow/compiler/mlir/lite/kernels/internal:compatibility_macros",
        "//tensorflow/compiler/mlir/lite/schema:schema_fbs_with_mutable",
        "//tensorflow/compiler/mlir/lite/schema:schema_utils",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
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
        "//tensorflow/compiler/mlir/lite/core/api",
        "//tensorflow/compiler/mlir/lite/core/c:c_api_types",
        "//tensorflow/compiler/mlir/lite/core/c:common",
        "//tensorflow/compiler/mlir/lite/kernels:kernel_util",
        "//tensorflow/compiler/mlir/lite/schema:schema_fbs",
        "//tensorflow/compiler/mlir/lite/schema:schema_utils",
        "@flatbuffers//:runtime_cc",
    ],
)

# LINT.IfChange(gpu_compatibility)
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
        "//tensorflow/compiler/mlir/lite:builtin_ops",
        "//tensorflow/compiler/mlir/lite/core/c:builtin_op_data",
        "//tensorflow/compiler/mlir/lite/core/c:common",
        "//tensorflow/compiler/mlir/lite/schema:schema_fbs",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)
# LINT.ThenChange(//tensorflow/lite/tools/versioning:gpu_compatibility)

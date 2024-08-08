load("@flatbuffers//:build_defs.bzl", "flatbuffer_cc_library")
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load("//tensorflow/lite:special_rules.bzl", "tflite_portable_test_suite", "tflite_schema_utils_friends")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

filegroup(
    name = "tflite_internal_cc_3p_api_deps_src",
    srcs = [
        ":schema_fbs_srcs",
        ":schema_utils.h",
    ],
    visibility = [
        "//tensorflow/lite:__pkg__",
    ],
)

# This is the package group declaration to which targets for TensorFlow Lite
# Flatbuffer schema utilities.
#
# Its usage should be rare, and is often abused by tools that are doing
# Flatbuffer creation/manipulation in unofficially supported ways.
package_group(
    name = "utils_friends",
    packages = [
        "//tensorflow/compiler/mlir/lite/...",
        "//tensorflow/lite/...",
    ] + tflite_schema_utils_friends(),
)

exports_files([
    "conversion_metadata.fbs",
])

flatbuffer_cc_library(
    name = "schema_fbs",
    srcs = ["//tensorflow/compiler/mlir/lite/schema:schema.fbs"],
    compatible_with = get_compatible_with_portable(),
)

# Generic schema for flatbuffer converter (but with mutable makes bigger).
flatbuffer_cc_library(
    name = "schema_fbs_with_mutable",
    srcs = ["//tensorflow/compiler/mlir/lite/schema:schema.fbs"],
    compatible_with = get_compatible_with_portable(),
    flatc_args = [
        "--gen-mutable",
        "--gen-object-api",
    ],
    out_prefix = "mutable/",
)

# Generic schema for inference on device (but with reflections makes bigger).
flatbuffer_cc_library(
    name = "schema_fbs_with_reflection",
    srcs = ["//tensorflow/compiler/mlir/lite/schema:schema.fbs"],
    compatible_with = get_compatible_with_portable(),
    flatc_args = [
        "--reflect-types",
        "--reflect-names",
        "--no-union-value-namespacing",
        "--gen-object-api",
    ],
    out_prefix = "reflection/",
)

cc_library(
    name = "schema_utils",
    hdrs = ["schema_utils.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:public"],
    deps = ["//tensorflow/compiler/mlir/lite/schema:schema_utils"],
)

cc_library(
    name = "schema_conversion_utils",
    hdrs = ["schema_conversion_utils.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:public"],
    deps = ["//tensorflow/compiler/mlir/lite/schema:schema_conversion_utils"],
)

flatbuffer_cc_library(
    name = "conversion_metadata_fbs",
    srcs = ["conversion_metadata.fbs"],
    compatible_with = get_compatible_with_portable(),
)

tflite_portable_test_suite()

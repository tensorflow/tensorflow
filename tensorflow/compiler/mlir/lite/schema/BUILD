load("@flatbuffers//:build_defs.bzl", "flatbuffer_cc_library")
load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

exports_files(
    srcs = [
        "conversion_metadata.fbs",
        "debug_metadata.fbs",
        "schema.fbs",
        "schema_v0.fbs",
        "schema_v1.fbs",
        "schema_v2.fbs",
        "schema_v3.fbs",
        "schema_v3a.fbs",
        "schema_v3b.fbs",
    ],
)

filegroup(
    name = "tflite_internal_cc_3p_api_deps_src",
    srcs = [
        ":debug_metadata_fbs_srcs",
        ":schema_fbs_srcs",
        ":schema_utils.h",
    ],
    visibility = ["//tensorflow/lite:__pkg__"],
)

flatbuffer_cc_library(
    name = "schema_fbs",
    srcs = ["schema.fbs"],
    compatible_with = get_compatible_with_portable(),
)

# Generic schema for flatbuffer converter (but with mutable makes bigger).
flatbuffer_cc_library(
    name = "schema_fbs_with_mutable",
    srcs = ["schema.fbs"],
    compatible_with = get_compatible_with_portable(),
    flatc_args = [
        "--gen-mutable",
        "--gen-object-api",
    ],
    out_prefix = "mutable/",
)

flatbuffer_cc_library(
    name = "debug_metadata_fbs",
    srcs = ["debug_metadata.fbs"],
    compatible_with = get_compatible_with_portable(),
)

flatbuffer_cc_library(
    name = "debug_metadata_fbs_with_mutable",
    srcs = ["debug_metadata.fbs"],
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
    srcs = ["schema.fbs"],
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
    srcs = ["schema_utils.cc"],
    hdrs = ["schema_utils.h"],
    compatible_with = get_compatible_with_portable(),
    # visibility = [":utils_friends"],
    deps = [
        ":schema_fbs",
        "//tensorflow/compiler/mlir/lite/kernels/internal:compatibility_macros",
        "@flatbuffers//:runtime_cc",
    ],
)

cc_library(
    name = "schema_conversion_utils",
    srcs = ["schema_conversion_utils.cc"],
    hdrs = ["schema_conversion_utils.h"],
    compatible_with = get_compatible_with_portable(),
    # visibility = [":utils_friends"],
    deps = [
        ":schema_fbs",
        "@flatbuffers",
    ],
)

# Schema test to make sure we don't introduce backward incompatible changes
# to schemas.
tf_cc_test(
    name = "flatbuffer_compatibility_test",
    size = "small",
    srcs = ["flatbuffer_compatibility_test.cc"],
    data = [
        "schema.fbs",
        "schema_v3b.fbs",
    ],
    tags = [
        "no_oss",
        "tflite_not_portable_android",
        "tflite_not_portable_ios",
    ],
    deps = [
        "//tensorflow/core/platform",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers//:flatc_library",
    ],
)

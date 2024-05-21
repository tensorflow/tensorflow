load("@flatbuffers//:build_defs.bzl", "flatbuffer_cc_library")
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

exports_files(
    srcs = ["schema.fbs"],
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

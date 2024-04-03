load("@flatbuffers//:build_defs.bzl", "flatbuffer_cc_library")
load("//tensorflow:strict.default.bzl", "py_strict_binary", "py_strict_library", "py_strict_test")
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
        ":schema_utils.cc",
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

py_strict_binary(
    name = "upgrade_schema",
    srcs = ["upgrade_schema.py"],
    python_version = "PY3",
    deps = [":upgrade_schema_main_lib"],
)

py_strict_library(
    name = "upgrade_schema_main_lib",
    srcs = [
        "upgrade_schema.py",
    ],
    data = [
        "schema_v0.fbs",
        "schema_v1.fbs",
        "schema_v2.fbs",
        "schema_v3.fbs",
        "@flatbuffers//:flatc",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/platform:resource_loader",
    ],
)

py_strict_test(
    name = "upgrade_schema_test",
    size = "small",
    srcs = ["upgrade_schema_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "manual",
        "no_oss",
        "no_pip",
        "notap",
    ],
    deps = [
        ":upgrade_schema_main_lib",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
    ],
)

# copybara:uncomment_begin(google-only)
# py_strict_test(
#     name = "schema_validation_test",
#     srcs = ["schema_validation_test.py"],
#     data = [
#         ":schema_fbs_srcs",
#         ":schema_generated.h.oss",
#     ],
#     python_version = "PY3",
#     # TODO(b/235550563): Enable this TAP with FlatBuffer 2.0.6 migration.
#     tags = [
#         "manual",
#         "notap",
#     ],
#     deps = [
#         "//testing/pybase",
#         "@absl_py//absl/flags",
#     ],
# )
# copybara:uncomment_end

exports_files([
    "conversion_metadata.fbs",
    "schema.fbs",
    "schema_v0.fbs",
    "schema_v1.fbs",
    "schema_v2.fbs",
    "schema_v3.fbs",
    "schema_v3a.fbs",
    "schema_v3b.fbs",
])

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

# Schema test to make sure we don't introduce backward incompatible changes
# to schemas.
cc_test(
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

cc_library(
    name = "schema_utils",
    srcs = ["schema_utils.cc"],
    hdrs = ["schema_utils.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:public"],
    deps = [
        ":schema_fbs",
        "//tensorflow/lite/kernels/internal:compatibility",
        "@flatbuffers//:runtime_cc",
    ],
)

cc_library(
    name = "schema_conversion_utils",
    srcs = ["schema_conversion_utils.cc"],
    hdrs = ["schema_conversion_utils.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:public"],
    deps = [
        ":schema_fbs",
        "//tensorflow/lite/kernels/internal:compatibility",
        "@flatbuffers",
    ],
)

flatbuffer_cc_library(
    name = "conversion_metadata_fbs",
    srcs = ["conversion_metadata.fbs"],
    compatible_with = get_compatible_with_portable(),
)

tflite_portable_test_suite()

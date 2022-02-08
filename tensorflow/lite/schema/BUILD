load("//tensorflow:tensorflow.bzl", "py_test")
load("//tensorflow/lite:special_rules.bzl", "tflite_portable_test_suite", "tflite_schema_utils_friends")
load("@flatbuffers//:build_defs.bzl", "flatbuffer_cc_library")
load("//tensorflow:tensorflow.bzl", "get_compatible_with_portable")

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

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

py_binary(
    name = "upgrade_schema",
    srcs = ["upgrade_schema.py"],
    python_version = "PY3",
    deps = [":upgrade_schema_main_lib"],
)

py_library(
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
        "//tensorflow/python:platform",
    ],
)

py_test(
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
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_test_lib",
    ],
)

# copybara:uncomment_begin(google-only)
# py_test(
#     name = "schema_validation_test",
#     srcs = ["schema_validation_test.py"],
#     data = [
#         "//tensorflow/lite/schema:schema_fbs_srcs",
#         "//tensorflow/lite/schema:schema_generated.h.oss",
#     ],
#     python_version = "PY3",
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
    visibility = [":utils_friends"],
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
    visibility = [":utils_friends"],
    deps = [
        ":schema_fbs",
        "//tensorflow/lite/kernels/internal:compatibility",
        "@flatbuffers",
    ],
)

tflite_portable_test_suite()

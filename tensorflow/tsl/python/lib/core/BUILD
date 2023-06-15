# Description:
# Implementation of custom numpy floats.
load("//tensorflow/tsl:tsl.bzl", "if_windows")
load("//tensorflow/tsl:tsl.default.bzl", "tsl_pybind_extension")
load("//tensorflow/tsl/platform:rules_cc.bzl", "cc_shared_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    features = [
        # For bfloat16.so (see b/259896740)
        "windows_export_all_symbols",
    ],
    licenses = ["notice"],
)

filegroup(
    name = "numpy_hdr",
    srcs = ["numpy.h"],
)

filegroup(
    name = "basic_hdrs",
    srcs = [
        "numpy.h",
    ],
)

cc_library(
    name = "bfloat16_lib",
    srcs = [
        "bfloat16.cc",
    ],
    hdrs = [
        "bfloat16.h",
    ],
    deps = [
        ":custom_float",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:types",
        "//tensorflow/tsl/python/lib/core:numpy",
        "//third_party/eigen3",
        "//third_party/python_runtime:headers",  # build_cleaner: keep; DNR: b/35864863
    ],
)

cc_shared_library(
    name = "bfloat16.so",
    roots = [":bfloat16_lib"],
    # TODO(tlongeri): If this is not explicitly specified, dependent DLLs will expect "bfloat16.so"
    # and we will generate "bfloat16.so.dll", for some reason
    shared_lib_name = if_windows("bfloat16.so", None),
    static_deps = [
        # TODO(ddunleavy): If cc_shared_library is ever not a noop in g3, change
        # this to be more specific.
        "//:__subpackages__",
        "@//:__subpackages__",
        "@com_google_absl//:__subpackages__",
        # TODO(tlongeri): Ugly work-around to avoid reverse copybara check breaking. Change to
        # just "//third_party/eigen3" after OSS TF removes //third_party/eigen3 in favor of
        # @eigen_archive
        "@" + "eigen_archive//:__subpackages__",
        "//third_party/" + "eigen3:__subpackages__",
        "@local_config_python//:__subpackages__",
        "@nsync//:__subpackages__",
    ],
)

tsl_pybind_extension(
    name = "pywrap_bfloat16",
    srcs = ["bfloat16_wrapper.cc"],
    dynamic_deps = [":bfloat16.so"],
    static_deps = [
        "@pybind11//:__subpackages__",
        "@local_config_python//:__subpackages__",
    ],
    deps = [
        ":bfloat16_lib",
        "//third_party/python_runtime:headers",
        "@pybind11",
    ],
)

cc_library(
    name = "custom_float",
    hdrs = [
        "custom_float.h",
    ],
    deps = [
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/python/lib/core:numpy",
        "//third_party/eigen3",
        "//third_party/python_runtime:headers",  # build_cleaner: keep; DNR: b/35864863
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "float8_hdr",
    hdrs = [
        "float8.h",
    ],
    deps = [
        "//third_party/python_runtime:headers",  # build_cleaner: keep; DNR: b/35864863
    ],
)

cc_library(
    name = "float8_lib",
    srcs = [
        "float8.cc",
    ],
    hdrs = [
        "float8.h",
    ],
    deps = [
        ":custom_float",
        ":numpy",
        "//tensorflow/tsl/platform:float8",
        "//tensorflow/tsl/platform:types",
        "//third_party/eigen3",
        "//third_party/python_runtime:headers",  # build_cleaner: keep; DNR: b/35864863
    ],
)

cc_library(
    name = "custom_casts_lib",
    srcs = ["custom_casts.cc"],
    hdrs = ["custom_casts.h"],
    deps = [
        ":bfloat16_lib",
        ":float8_lib",
        "//tensorflow/tsl/platform:types",
        "//tensorflow/tsl/python/lib/core:numpy",
        "//third_party/python_runtime:headers",  # build_cleaner: keep; DNR: b/35864863
    ],
)

cc_library(
    name = "custom_casts_hdr",
    hdrs = [
        "custom_casts.h",
    ],
)

cc_library(
    name = "numpy",
    srcs = ["numpy.cc"],
    hdrs = ["numpy.h"],
    deps = [
        "//third_party/py/numpy:headers",
        "//third_party/python_runtime:headers",
    ],
)

# Directory-level target.
cc_library(
    name = "core",
    deps = [
        ":bfloat16_lib",
        ":custom_casts_lib",
        ":float8_lib",
        ":numpy",
    ],
)

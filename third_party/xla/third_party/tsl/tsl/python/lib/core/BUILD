# Description:
#   Implementation of custom numpy floats.

load("//tsl:tsl.bzl", "if_windows")
load("//tsl:tsl.default.bzl", "tsl_pybind_extension")
load("//tsl/platform:rules_cc.bzl", "cc_shared_library")

package(
    default_visibility = ["//visibility:public"],
    features = [
        # For ml_dtypes.so (see b/259896740)
        "windows_export_all_symbols",
    ],
    licenses = ["notice"],
)

filegroup(
    name = "numpy_hdr",
    srcs = ["numpy.h"],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "basic_hdrs",
    srcs = ["numpy.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "ml_dtypes_lib",
    srcs = ["ml_dtypes.cc"],
    hdrs = ["ml_dtypes.h"],
    # Requires data dependency in order to load py module from C++.
    data = [
        "@ml_dtypes",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//third_party/python_runtime:headers",  # build_cleaner: keep; DNR: b/35864863
        "//tsl/python/lib/core:numpy",
        "@pybind11",
    ],
)

cc_shared_library(
    name = "ml_dtypes.so",
    roots = [":ml_dtypes_lib"],
    # TODO(tlongeri): If this is not explicitly specified, dependent DLLs will expect "ml_dtypes.so"
    # and we will generate "ml_dtypes.so.dll", for some reason
    shared_lib_name = if_windows("ml_dtypes.so", None),
    static_deps = [
        # TODO(ddunleavy): If cc_shared_library is ever not a noop in g3, change
        # this to be more specific.
        "//:__subpackages__",
        "@//:__subpackages__",
        "@com_google_absl//:__subpackages__",
        "@local_config_python//:__subpackages__",
        "@pybind11//:__subpackages__",
        "@nsync//:__subpackages__",
        "//tsl:__subpackages__",
    ],
)

tsl_pybind_extension(
    name = "pywrap_ml_dtypes",
    srcs = ["ml_dtypes_wrapper.cc"],
    dynamic_deps = [":ml_dtypes.so"],
    static_deps = [
        "@//:__subpackages__",
        "@pybind11//:__subpackages__",
        "@local_config_python//:__subpackages__",
    ],
    deps = [
        ":ml_dtypes_lib",
        "//third_party/python_runtime:headers",
        "@pybind11",
    ],
)

cc_library(
    name = "numpy",
    srcs = ["numpy.cc"],
    hdrs = ["numpy.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//third_party/py/numpy:headers",
        "//third_party/python_runtime:headers",
    ],
)

# Directory-level target.
cc_library(
    name = "core",
    visibility = ["//visibility:public"],
    deps = [
        ":ml_dtypes_lib",
        ":numpy",
    ],
)

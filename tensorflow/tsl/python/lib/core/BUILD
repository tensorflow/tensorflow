# Description:
# Implementation of custom numpy floats.

package(
    default_visibility = [
        "//visibility:public",
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
    name = "bfloat16_hdr",
    hdrs = [
        "bfloat16.h",
    ],
    deps = [
        "//third_party/python_runtime:headers",  # build_cleaner: keep; DNR: b/35864863
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

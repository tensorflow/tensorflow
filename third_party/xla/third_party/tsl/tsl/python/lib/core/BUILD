# Description:
#   Implementation of custom numpy floats.

package(
    default_visibility = ["//visibility:public"],
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
    copts = ["-fexceptions"],  # Required for pybind11.
    # Requires data dependency in order to load py module from C++.
    data = [
        "@ml_dtypes",
    ],
    features = [
        "-use_header_modules",  # Required for pybind11.
        "-parse_headers",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":numpy",
        "//third_party/py/numpy:headers",
        "//third_party/python_runtime:headers",  # build_cleaner: keep; DNR: b/35864863
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/synchronization",
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

# Description:
#   Implementation of custom numpy floats.

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
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
    srcs = ["numpy.h"],
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
    deps = [
        ":numpy",
        "//third_party/py/numpy:headers",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/synchronization",
        "@local_config_python//:python_headers",  # build_cleaner: keep; DNR: b/35864863
        "@pybind11",
    ],
)

cc_library(
    name = "numpy",
    srcs = ["numpy.cc"],
    hdrs = ["numpy.h"],
    deps = [
        "//third_party/py/numpy:headers",
        "@local_config_python//:python_headers",
    ],
)

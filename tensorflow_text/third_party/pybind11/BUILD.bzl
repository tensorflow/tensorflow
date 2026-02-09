"""
BUILD file for pybind11 package, since the github version does not have one.
"""

load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "pybind11",
    hdrs = glob(
        include = [
            "include/pybind11/*.h",
            "include/pybind11/detail/*.h",
        ],
        exclude = [
            "include/pybind11/common.h",
            "include/pybind11/eigen.h",
        ],
    ),
    copts = [
        "-fexceptions",
        "-Wno-undefined-inline",
        "-Wno-pragma-once-outside-header",
    ],
    includes = ["include"],
    strip_include_prefix = "include",
    deps = [
        "@local_xla//third_party/python_runtime:headers",
    ],
)

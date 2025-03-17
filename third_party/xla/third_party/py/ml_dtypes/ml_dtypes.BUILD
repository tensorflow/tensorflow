""" Main ml_dtypes library. """

load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "float8",
    hdrs = ["include/float8.h"],
    deps = ["@eigen_archive//:eigen3"],
)

cc_library(
    name = "intn",
    hdrs = ["include/intn.h"],
)

cc_library(
    name = "mxfloat",
    hdrs = ["include/mxfloat.h"],
    deps = [
        ":float8",
        "@eigen_archive//:eigen3",
    ],
)

pybind_extension(
    name = "_ml_dtypes_ext",
    srcs = [
        "_src/common.h",
        "_src/custom_float.h",
        "_src/dtypes.cc",
        "_src/intn_numpy.h",
        "_src/numpy.cc",
        "_src/numpy.h",
        "_src/ufuncs.h",
    ],
    visibility = [":__subpackages__"],
    deps = [
        ":float8",
        ":intn",
        ":mxfloat",
        "@eigen_archive//:eigen3",
        "@local_xla//third_party/py/numpy:headers",
    ],
)

py_library(
    name = "ml_dtypes",
    srcs = [
        "__init__.py",
        "_finfo.py",
        "_iinfo.py",
    ],
    imports = ["."],  # Import relative to _this_ directory, not the root.
    deps = [":_ml_dtypes_ext"],
)

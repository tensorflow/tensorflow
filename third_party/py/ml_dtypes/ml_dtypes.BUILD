load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

exports_files(["LICENSE"])

cc_library(
    name = "float8",
    hdrs = ["include/float8.h"],
    include_prefix = "ml_dtypes",
    # Internal headers are all relative to . but other packages
    # include these headers with the prefix.
    includes = [
        ".",
        "ml_dtypes",
    ],
    deps = ["@eigen_archive//:eigen3"],
)

cc_library(
    name = "intn",
    hdrs = ["include/intn.h"],
    include_prefix = "ml_dtypes",
    # Internal headers are all relative to . but other packages
    # include these headers with the  prefix.
    includes = [
        ".",
        "ml_dtypes",
    ],
)

pybind_extension(
    name = "_ml_dtypes_ext",
    srcs = [
        "_src/common.h",
        "_src/custom_float.h",
        "_src/dtypes.cc",
        "_src/int4_numpy.h",
        "_src/numpy.cc",
        "_src/numpy.h",
        "_src/ufuncs.h",
    ],
    includes = ["ml_dtypes"],
    visibility = [":__subpackages__"],
    deps = [
        ":float8",
        ":intn",
        "@eigen_archive//:eigen3",
        "@local_tsl//third_party/py/numpy:headers",
    ],
)

py_library(
    name = "ml_dtypes",
    srcs = [
        "__init__.py",
        "_finfo.py",
        "_iinfo.py",
    ],
    deps = [":_ml_dtypes_ext"],
)

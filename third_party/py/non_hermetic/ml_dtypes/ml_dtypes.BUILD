load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

exports_files(["LICENSE"])

cc_library(
    name = "float8",
    hdrs = ["include/float8.h"],
    # Internal headers are all relative to , but other packages
    # include these headers with the  prefix.
    includes = [
        ".",
        "ml_dtypes",
    ],
    deps = ["@org_tensorflow//third_party/eigen3"],
)

pybind_extension(
    name = "_custom_floats",
    srcs = [
        "_src/common.h",
        "_src/custom_float.h",
        "_src/dtypes.cc",
        "_src/int4.h",
        "_src/numpy.cc",
        "_src/numpy.h",
        "_src/ufuncs.h",
    ],
    includes = ["ml_dtypes"],
    visibility = [":__subpackages__"],
    deps = [
        ":float8",
        "@org_tensorflow//third_party/eigen3",
        "@org_tensorflow//third_party/py/numpy:headers",
    ],
)

py_library(
    name = "ml_dtypes",
    srcs = [
        "__init__.py",
        "_finfo.py",
        "_iinfo.py",
    ],
    deps = [":_custom_floats"],
)

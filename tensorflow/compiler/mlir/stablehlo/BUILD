load("//tensorflow:strict.default.bzl", "py_strict_test")
load("//tensorflow:pytype.default.bzl", "pytype_strict_library")
load("@local_tsl//tsl:tsl.default.bzl", "tsl_pybind_extension")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        ":friends",
    ],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    packages = [
        "//tensorflow/compiler/tests/...",
    ],
)

tsl_pybind_extension(
    name = "stablehlo_extension",
    srcs = [
        "stablehlo.cc",
        "@stablehlo//:stablehlo/integrations/python/PortableApi.cpp",
    ],
    hdrs = [
        "@stablehlo//:stablehlo/integrations/python/PortableApi.h",
    ],
    copts = [
        "-fexceptions",
        "-frtti",
    ],
    features = ["-use_header_modules"],
    deps = [
        "@pybind11",
        "@stablehlo//:stablehlo_portable_api",
    ],
)

pytype_strict_library(
    name = "stablehlo",
    srcs = ["stablehlo.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":stablehlo_extension",
    ],
)

py_strict_test(
    name = "stablehlo_test",
    srcs = ["stablehlo_test.py"],
    python_version = "PY3",
    deps = [
        ":stablehlo",
    ],
)

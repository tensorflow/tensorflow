load("//xla:py_strict.bzl", "py_strict_test")
load("//xla/tsl:tsl.default.bzl", "tsl_pybind_extension")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

# copybara:uncomment package(default_applicable_licenses = ["//tensorflow:license"])

# This file is a little messy in order to resolve some unfortunate issues
cc_library(
    name = "profile_data_lib",
    srcs = ["profile_data_lib.cc"],
    hdrs = ["profile_data_lib.h"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    visibility = [
        "//xla/python:__pkg__",
    ],
    deps = [
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:protobuf",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@nanobind",
    ],
    alwayslink = 1,
)

cc_library(
    name = "profile_data_module",
    srcs = ["build_profile_data_module.cc"],
    hdrs = ["build_profile_data_module.h"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    visibility = [
        "//third_party/py/jax/jaxlib/xla:__pkg__",
        "//visibility:private",  #TODO(zcrank): remove when "profile_data" target is removed.
        "//xla/python:__pkg__",
    ],
    deps = [
        ":profile_data_lib",
        "@nanobind",
    ],
)

# This target builds a standalone module for profile_data.py, which some existing code depends on.
# Please do not add new dependencies on this target.
# TODO(zcrank): remove this target once the existing code (particularly Waymo) is updated.
tsl_pybind_extension(
    name = "profile_data",
    srcs = ["nb_module_profile_data.cc"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    pytype_srcs = ["profile_data.pyi"],
    visibility = ["//visibility:public"],
    deps = [
        ":profile_data_module",
        "@nanobind",
    ],
)

py_strict_test(
    name = "profile_data_test",
    srcs = ["profile_data_test.py"],
    tags = [
        "no_oss",  # TODO(phawkins): This test passes, but requires --config=monolithic.
        "optonly",
    ],
    deps = [
        ":profile_data",
        "@absl_py//absl/testing:absltest",
    ],
)

load("//xla:strict.default.bzl", "py_strict_test")
load("//xla/tsl:tsl.default.bzl", "tsl_pybind_extension")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

# copybara:uncomment package(default_applicable_licenses = ["//tensorflow:license"])

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

# This target builds a standalone module for profile_data.py, which some existing code depends on.
tsl_pybind_extension(
    name = "profile_data",
    srcs = ["py_profile_data.cc"],
    hdrs = ["py_profile_data.h"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":profile_data_lib",
        "@local_tsl//tsl/platform:protobuf",
        "@local_tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@nanobind",
    ],
)

# We need a separate target to build the python profile data submodule for xla_extension
cc_library(
    name = "profile_data_submodule",
    srcs = ["py_profile_data.cc"],
    hdrs = ["py_profile_data.h"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    visibility = ["//visibility:public"],
    deps = [
        ":profile_data_lib",
        "@local_tsl//tsl/platform:protobuf",
        "@local_tsl//tsl/profiler/protobuf:xplane_proto_cc",
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
        # copybara:uncomment "//pyglib:gfile",
        "@absl_py//absl/testing:absltest",
        "//xla/python:xla_extension",
    ],
)

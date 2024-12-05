load("//xla:pytype.default.bzl", "pytype_strict_library")
load("//xla:strict.default.bzl", "py_strict_test")
load("//xla/tsl:tsl.default.bzl", "tsl_pybind_extension")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//xla:friends",
    ],
)

cc_library(
    name = "kernel_runner",
    srcs = ["kernel_runner.cc"],
    hdrs = ["kernel_runner.h"],
    deps = [
        "//xla:literal",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:logging",
    ],
)

tsl_pybind_extension(
    name = "kernel_runner_extention",
    srcs = ["kernel_runner_extention.cc"],
    deps = [
        ":kernel_runner",
        "//xla:literal",
        "//xla/codegen:kernel_emitter",
        "//xla/codegen:kernel_spec",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_config_python//:python_headers",
        "@nanobind",
    ],
)

pytype_strict_library(
    name = "kernel_runner_pylib",
    srcs = ["kernel_runner.py"],
    srcs_version = "PY3",
    deps = [
        ":kernel_runner_extention",
        "//third_party/py/numpy",
        "//xla/python:xla_extension",
    ],
)

py_strict_test(
    name = "kernel_runner_test_py",
    srcs = ["kernel_runner_test.py"],
    main = "kernel_runner_test.py",
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":kernel_runner_pylib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:absltest",
    ],
)

load("//xla:py_strict.bzl", "py_strict_test")
load("//xla:pytype.bzl", "pytype_strict_library")
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
    testonly = 1,
    srcs = ["kernel_runner.cc"],
    hdrs = ["kernel_runner.h"],
    deps = [
        "//xla:literal",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:logging",
    ],
)

tsl_pybind_extension(
    name = "_extension",
    testonly = 1,
    srcs = ["kernel_runner_extension.cc"],
    visibility = ["//visibility:private"],  # the extension should always be linked via testlib
    deps = [
        ":kernel_runner",
        # placeholder for index annotation deps  # buildcleaner: keep
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@nanobind",
        "@local_config_python//:python_headers",  # buildcleaner: keep
        "//xla:comparison_util",
        "//xla:debug_options_flags",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/codegen:kernel_definition",
        "//xla/codegen:kernel_emitter",
        "//xla/codegen:kernel_source",
        "//xla/codegen:kernel_spec",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/python:nb_absl_inlined_vector",
        "//xla/python:nb_absl_span",
        "//xla/service:buffer_assignment",
        "//xla/service:hlo_module_config",
    ],
)

pytype_strict_library(
    name = "testlib",
    testonly = 1,
    srcs = [
        "__init__.py",
        "utilities.py",
    ],
    deps = [
        ":_extension",
        "//third_party/py/numpy",
        "//xla/python:xla_extension",
        "@ml_dtypes_py//ml_dtypes",  # buildcleaner: keep (transitively depend on it via xla_extension)
    ],
)

py_strict_test(
    name = "kernel_runner_test_py",
    srcs = ["kernel_runner_test.py"],
    main = "kernel_runner_test.py",
    tags = [
        "no_oss",
    ],
    deps = [
        ":_extension",
        ":testlib",
        "//third_party/py/numpy",
        "//xla/python:xla_extension",
        "@absl_py//absl/testing:absltest",
    ],
)

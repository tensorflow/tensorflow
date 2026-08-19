load("@xla//third_party/rules_python/python:py_test.bzl", "py_test")
load("//xla:pytype.bzl", "pytype_strict_binary", "pytype_strict_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

pytype_strict_library(
    name = "check_contents",
    srcs = ["check_contents.py"],
    deps = [":diff_parser"],
)

pytype_strict_library(
    name = "check_header_guards",
    srcs = ["check_header_guards.py"],
)

pytype_strict_library(
    name = "diff_parser",
    srcs = ["diff_parser.py"],
    visibility = ["//visibility:public"],
)

pytype_strict_library(
    name = "generate_compile_commands",
    srcs = ["generate_compile_commands.py"],
)

py_test(
    name = "check_contents_test",
    srcs = ["check_contents_test.py"],
    data = [
        "testdata/bad_cc.diff",
        "testdata/important_cc.diff",
    ],
    deps = [
        ":check_contents",
        ":diff_parser",
        "//build_tools:test_utils",
        "@absl_py//absl/testing:absltest",
    ],
)

py_test(
    name = "diff_parser_test",
    srcs = ["diff_parser_test.py"],
    data = [
        "testdata/bad_cc.diff",
        "testdata/crosstool.diff",
        "testdata/important_cc.diff",
    ],
    deps = [
        ":diff_parser",
        "//build_tools:test_utils",
        "@absl_py//absl/testing:absltest",
    ],
)

py_test(
    name = "generate_compile_commands_test",
    srcs = ["generate_compile_commands_test.py"],
    deps = [
        ":generate_compile_commands",
        "@absl_py//absl/testing:absltest",
    ],
)

py_test(
    name = "check_header_guards_test",
    srcs = ["check_header_guards_test.py"],
    deps = [
        ":check_header_guards",
        "@absl_py//absl/testing:absltest",
    ],
)

pytype_strict_binary(
    name = "tags",
    srcs = ["tags.py"],
)

load("//tensorflow:pytype.default.bzl", "pytype_strict_library")
load("//tensorflow:strict.default.bzl", "py_strict_test")
load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("//tensorflow:tensorflow.default.bzl", "filegroup")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

exports_files(
    ["flags_api_wrapper.cc"],
    visibility = ["//tensorflow:internal"],
)

cc_library(
    name = "flags_headers",
    textual_hdrs = [
        "flag_defs.h",
        "flags.h",
    ],
    visibility = ["//tensorflow:internal"],
    deps = ["//tensorflow/core/platform:stringpiece"],
)

filegroup(
    name = "flags_headers_filegroup",
    srcs = [
        "flag_defs.h",
        "flags.h",
    ],
    visibility = ["//tensorflow:internal"],
)

filegroup(
    name = "mobile_srcs_only_runtime",
    srcs = [
        "flag_defs.cc",
        "flag_defs.h",
        "flags.cc",
        "flags.h",
    ],
    visibility = ["//tensorflow:internal"],
)

cc_library(
    name = "flag_defs",
    srcs = ["flag_defs.cc"],
    hdrs = ["flag_defs.h"],
    visibility = ["//tensorflow:internal"],
    deps = [
        ":flags",
    ],
)

cc_library(
    name = "flags",
    srcs = ["flags.cc"],
    hdrs = ["flags.h"],
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/core/platform:stringpiece",
        "//tensorflow/core/util:env_var",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "flags_test",
    srcs = ["flags_test.cc"],
    deps = [
        ":flag_defs",
        ":flags",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

pytype_strict_library(
    name = "flags_py",
    srcs = ["flags.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/python:flags_pybind",
        "//tensorflow/python:pywrap_tensorflow",
    ],
)

py_strict_test(
    name = "flags_py_test",
    size = "small",
    srcs = ["flags_test.py"],
    main = "flags_test.py",
    python_version = "PY3",
    deps = [
        ":flags_py",
        #internal proto upb dep
        "//tensorflow/python/platform:client_testlib",
    ],
)

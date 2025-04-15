load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load("//tensorflow:pytype.default.bzl", "pytype_strict_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow/python/tools/api/generator2:__subpackages__"],
    licenses = ["notice"],
)

config_setting(
    name = "static_gen",
    define_values = {
        "TF_API_INIT_LOADING": "static",
    },
)

bzl_library(
    name = "apis_bzl",
    srcs = ["apis.bzl"],
    visibility = ["//visibility:private"],
    deps = [":patterns_bzl"],
)

bzl_library(
    name = "generate_api_bzl",
    srcs = ["generate_api.bzl"],
    deps = [
        ":apis_bzl",
        ":patterns_bzl",
        "//tensorflow/python/tools/api/generator:api_init_files",
        "@bazel_skylib//lib:paths",
    ],
)

bzl_library(
    name = "patterns_bzl",
    srcs = ["patterns.bzl"],
    visibility = ["//visibility:private"],
)

pytype_strict_library(
    name = "docstrings",
    srcs = ["docstrings.py"],
    visibility = ["//visibility:public"],
)

load("@bazel_skylib//:bzl_library.bzl", "bzl_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow/python/tools/api/generator2:__subpackages__"],
    licenses = ["notice"],
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
    ],
)

bzl_library(
    name = "patterns_bzl",
    srcs = ["patterns.bzl"],
    visibility = ["//visibility:private"],
)

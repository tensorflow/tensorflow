load("//tools/build_defs/testing:bzl_library.bzl", "bzl_library")

# copybara:uncomment package(default_applicable_licenses = ["//third_party/tensorflow:license"])

filegroup(
    name = "workspace",
    srcs = ["workspace.bzl"],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    visibility = ["//visibility:public"],
)

bzl_library(
    name = "workspace_bzl",
    srcs = ["workspace.bzl"],
    parse_tests = False,
    visibility = ["//visibility:private"],
)

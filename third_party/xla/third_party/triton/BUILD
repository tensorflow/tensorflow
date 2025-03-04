load("@bazel_skylib//:bzl_library.bzl", "bzl_library")

# copybara:uncomment_begin
# package(default_applicable_licenses = ["//tensorflow:license"])
#
# filegroup(
#     name = "patch_files",
#     srcs = glob([
#         "xla_extensions/**",
#         "llvm_integration/**",
#         "temporary/**",
#     ]),
#     visibility = ["//visibility:public"],
# )
# copybara:uncomment_end

filegroup(
    name = "workspace",
    srcs = ["workspace.bzl"],
    visibility = ["//visibility:public"],
)

bzl_library(
    name = "workspace_bzl",
    srcs = ["workspace.bzl"],
    parse_tests = False,
    visibility = ["//visibility:private"],
)

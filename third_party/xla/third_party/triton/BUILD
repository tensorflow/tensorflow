load("//tools/build_defs/testing:bzl_library.bzl", "bzl_library")

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
#     visibility = [
#         "//platforms/xla/experimental/tools/triton:__subpackages__",
#         "//third_party/triton:__subpackages__",
#     ],
# )
# copybara:uncomment_end

filegroup(
    name = "workspace",
    srcs = ["workspace.bzl"],
    visibility = ["//platforms/xla/experimental/tools/triton:__subpackages__"],
)

bzl_library(
    name = "workspace_bzl",
    srcs = ["workspace.bzl"],
    parse_tests = False,
    visibility = ["//visibility:private"],
)

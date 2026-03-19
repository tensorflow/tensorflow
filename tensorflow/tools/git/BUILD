# Description:
# Contains script to generate tensorflow/core/util/version_info.cc
# Also contains information about git repository deposited by configure
# in gen/...

load("@xla//third_party/rules_python/python:py_binary.bzl", "py_binary")
load("//tensorflow:tensorflow.bzl", "if_oss")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

py_binary(
    name = "gen_git_source",
    srcs = ["gen_git_source.py"],
    strict_deps = True,
    tags = if_oss(["no-remote-exec"]),
)

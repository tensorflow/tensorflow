# Description:
# Contains script to generate tensorflow/core/util/version_info.cc
# Also contains information about git repository deposited by configure
# in gen/...

load("//tensorflow:tensorflow.bzl", "if_oss")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

py_binary(
    name = "gen_git_source",
    srcs = ["gen_git_source.py"],
    exec_compatible_with = if_oss(["@local_execution_config_platform//:platform_constraint"]),
    python_version = "PY3",
    srcs_version = "PY3",
    tags = if_oss(["no-remote-exec"]),
)

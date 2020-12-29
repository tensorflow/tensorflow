# Description:
# Contains script to generate tensorflow/core/util/version_info.cc
# Also contains information about git repository deposited by configure
# in gen/...

package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],  # Apache 2.0
)

py_binary(
    name = "gen_git_source",
    srcs = ["gen_git_source.py"],
    exec_compatible_with = ["@local_execution_config_platform//:platform_constraint"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    tags = ["no-remote-exec"],
)

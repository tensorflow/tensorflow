# Description:
# Contains script to generate tensorflow/python/platform/build_info.py

package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],  # Apache 2.0
)

py_binary(
    name = "gen_build_info",
    srcs = ["gen_build_info.py"],
    exec_compatible_with = ["@local_execution_config_platform//:platform_constraint"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        "@six_archive//:six",
    ],
)

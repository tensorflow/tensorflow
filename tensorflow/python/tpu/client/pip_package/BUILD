# Description:
#  Tools for building the Cloud TPU Client pip package.

load("@rules_shell//shell:sh_binary.bzl", "sh_binary")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:private"],
)

licenses(["notice"])

sh_binary(
    name = "build_pip_package",
    srcs = ["build_pip_package.sh"],
    data = [
        "setup.py",
        "//tensorflow/python/tpu/client:client_lib",
    ],
)

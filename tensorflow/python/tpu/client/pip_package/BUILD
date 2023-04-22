# Description:
#  Tools for building the Cloud TPU Client pip package.

package(default_visibility = ["//visibility:private"])

licenses(["notice"])

sh_binary(
    name = "build_pip_package",
    srcs = ["build_pip_package.sh"],
    data = [
        "setup.py",
        "//tensorflow/python/tpu/client:client_lib",
    ],
)

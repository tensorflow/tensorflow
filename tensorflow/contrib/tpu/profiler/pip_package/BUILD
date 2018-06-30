# Description:
#  Tools for building the Cloud TPU profiler pip package.

package(default_visibility = ["//visibility:private"])

licenses(["notice"])  # Apache 2.0

sh_binary(
    name = "build_pip_package",
    srcs = ["build_pip_package.sh"],
    data = [
        "setup.py",
        "//tensorflow/contrib/tpu/profiler:capture_tpu_profile",
    ],
)

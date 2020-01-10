# Description:
# TensorFlow is a computational framework, primarily for use in machine
# learning applications.

package(
    licenses = ["notice"],  # Apache 2.0
)

sh_binary(
    name = "parallel_gpu_execute",
    srcs = ["parallel_gpu_execute.sh"],
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)

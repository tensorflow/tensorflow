# TensorFlow API backwards compatibility test goldens.

visibility = [
    "//tensorflow/tools/api:__subpackages__",
    # TODO(mihaimaruseac): Why do we need these for 1.13.2 patch?
    "//bazel_pip/tensorflow/tools:__subpackages__",
]

package(default_visibility = visibility)

licenses(["notice"])  # Apache 2.0

filegroup(
    name = "api_golden_v1",
    srcs = glob(["v1/*.pbtxt"]),
)

filegroup(
    name = "api_golden_v2",
    srcs = glob(["v2/*.pbtxt"]),
)

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def bazel_toolchains_archive():
    http_archive(
        name = "bazel_toolchains",
        sha256 = "77c2c3c562907a1114afde7b358bf3d5cc23dc61b3f2fd619bf167af0c9582a3",
        strip_prefix = "bazel-toolchains-dfc67056200b674accd08d8f9a21e328098c07e2",
        urls = [
            "http://mirror.tensorflow.org/github.com/bazelbuild/bazel-toolchains/archive/dfc67056200b674accd08d8f9a21e328098c07e2.tar.gz",
            "https://github.com/bazelbuild/bazel-toolchains/archive/dfc67056200b674accd08d8f9a21e328098c07e2.tar.gz",
        ],
    )

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def bazel_toolchains_archive():
    http_archive(
        name = "bazel_toolchains",
        sha256 = "88e818f9f03628eef609c8429c210ecf265ffe46c2af095f36c7ef8b1855fef5",
        strip_prefix = "bazel-toolchains-92dd8a7a518a2fb7ba992d47c8b38299fe0be825",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/bazelbuild/bazel-toolchains/archive/92dd8a7a518a2fb7ba992d47c8b38299fe0be825.tar.gz",
            "https://github.com/bazelbuild/bazel-toolchains/archive/92dd8a7a518a2fb7ba992d47c8b38299fe0be825.tar.gz",
        ],
    )

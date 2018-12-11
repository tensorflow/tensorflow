load("//tensorflow:version_check.bzl", "parse_bazel_version")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def bazel_toolchains_archive():
  if parse_bazel_version(native.bazel_version) >= parse_bazel_version("0.19"):
    # This version of the toolchains repo is incompatible with older bazel
    # versions - we can remove this once TensorFlow drops support for bazel
    # before 0.19.
    http_archive(
        name = "bazel_toolchains",
        sha256 = "41c48a189be489e2d15dec40e0057ea15b95ee5b39cc2a7e6cf663e31432c75e",
        strip_prefix = "bazel-toolchains-3f8c58fe530fedc446de04673bc1e32985887dea",
        urls = [
            "https://github.com/nlopezgi/bazel-toolchains/archive/3f8c58fe530fedc446de04673bc1e32985887dea.tar.gz",
        ],
    )
  else:
    http_archive(
        name = "bazel_toolchains",
        sha256 = "15b5858b1b5541ec44df31b94c3b8672815b31d71215a98398761ea9f4c4eedb",
        strip_prefix = "bazel-toolchains-6200b238c9c2d137c0d9a7262c80cc71d98e692b",
        urls = [
            "https://github.com/bazelbuild/bazel-toolchains/archive/6200b238c9c2d137c0d9a7262c80cc71d98e692b.tar.gz",
        ],
    )

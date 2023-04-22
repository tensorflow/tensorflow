"""Loads pasta python package."""

load("//third_party:repo.bzl", "tf_http_archive")

def repo():
    tf_http_archive(
        name = "pasta",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/pasta/archive/v0.1.8.tar.gz",
            "https://github.com/google/pasta/archive/v0.1.8.tar.gz",
        ],
        strip_prefix = "pasta-0.1.8",
        sha256 = "c6dc1118250487d987a7b1a404425822def2e8fb2b765eeebc96887e982b6085",
        build_file = "//third_party/pasta:BUILD.bazel",
        system_build_file = "//third_party/pasta:BUILD.system",
    )

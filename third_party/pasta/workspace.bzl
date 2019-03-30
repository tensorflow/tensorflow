"""Loads pasta python package."""

load("//third_party:repo.bzl", "third_party_http_archive")

def repo():
    third_party_http_archive(
        name = "pasta",
        urls = [
            "http://mirror.tensorflow.org/github.com/google/pasta/archive/v0.1.2.tar.gz",
            "https://github.com/google/pasta/archive/v0.1.2.tar.gz",
        ],
        strip_prefix = "pasta-0.1.2",
        sha256 = "53e4c009a5eac38e942deb48bfc2d3cfca62cd457255fa86ffedb7e40f726a0c",
        build_file = "//third_party/pasta:BUILD.bazel",
        system_build_file = "//third_party/pasta:BUILD.system",
    )

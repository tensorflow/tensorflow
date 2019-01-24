"""Loads pasta python package."""

load("//third_party:repo.bzl", "third_party_http_archive")

def repo():
    third_party_http_archive(
        name = "pasta",
        urls = [
            "https://mirror.bazel.build/github.com/google/pasta/archive/c3d72cdee6fc806251949e912510444d58d7413c.tar.gz",
            "https://github.com/google/pasta/archive/c3d72cdee6fc806251949e912510444d58d7413c.tar.gz",
        ],
        strip_prefix = "pasta-c3d72cdee6fc806251949e912510444d58d7413c",
        sha256 = "b5905f9cecc4b28363c563f3c4cb0545288bd35f7cc72c55066e97e53befc084",
        build_file = "//third_party/pasta:BUILD.bazel",
        system_build_file = "//third_party/pasta:BUILD.system",
    )

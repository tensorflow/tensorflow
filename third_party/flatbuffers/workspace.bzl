"""Loads the Flatbuffers library, used by TF Lite."""

load("//third_party:repo.bzl", "third_party_http_archive")

def repo():
    third_party_http_archive(
        name = "flatbuffers",
        strip_prefix = "flatbuffers-1.9.0",
        sha256 = "5ca5491e4260cacae30f1a5786d109230db3f3a6e5a0eb45d0d0608293d247e3",
        urls = [
            "https://mirror.bazel.build/github.com/google/flatbuffers/archive/v1.9.0.tar.gz",
            "https://github.com/google/flatbuffers/archive/v1.9.0.tar.gz",
        ],
        build_file = "//third_party/flatbuffers:BUILD.bazel",
        system_build_file = "//third_party/flatbuffers:BUILD.system",
        link_files = {
            "//third_party/flatbuffers:build_defs.bzl": "build_defs.bzl",
        },
    )

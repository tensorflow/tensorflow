"""Loads the Flatbuffers library, used by TF Lite."""

load("//third_party:repo.bzl", "tf_http_archive")

def repo():
    tf_http_archive(
        name = "flatbuffers",
        strip_prefix = "flatbuffers-1.12.0",
        sha256 = "62f2223fb9181d1d6338451375628975775f7522185266cd5296571ac152bc45",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/flatbuffers/archive/v1.12.0.tar.gz",
            "https://github.com/google/flatbuffers/archive/v1.12.0.tar.gz",
        ],
        build_file = "//third_party/flatbuffers:BUILD.bazel",
        system_build_file = "//third_party/flatbuffers:BUILD.system",
        link_files = {
            "//third_party/flatbuffers:build_defs.bzl": "build_defs.bzl",
        },
    )

"""Loads the Flatbuffers library, used by TF Lite."""

load("//third_party:repo.bzl", "third_party_http_archive")

def repo():
    third_party_http_archive(
        name = "flatbuffers",
        strip_prefix = "flatbuffers-a4b2884e4ed6116335d534af8f58a84678b74a17",
        sha256 = "6ff041dcaf873acbf0a93886e6b4f7704b68af1457e8b675cae88fbefe2de330",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/https://github.com/google/flatbuffers/archive/a4b2884e4ed6116335d534af8f58a84678b74a17.zip",
            "https://github.com/google/flatbuffers/archive/a4b2884e4ed6116335d534af8f58a84678b74a17.zip",
        ],
        build_file = "//third_party/flatbuffers:BUILD.bazel",
        system_build_file = "//third_party/flatbuffers:BUILD.system",
        delete = ["build_defs.bzl"],
        link_files = {
            "//third_party/flatbuffers:build_defs.bzl": "build_defs.bzl",
        },
    )

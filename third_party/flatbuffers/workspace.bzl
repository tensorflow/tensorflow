"""Loads the Flatbuffers library, used by TF Lite."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "flatbuffers",
        strip_prefix = "flatbuffers-23.1.20",
        sha256 = "3f074b30a3ea5a1c0fb23c208d2ab8db1281bdc4a1707ef2c049a1cc9cc13136",
        urls = tf_mirror_urls("https://github.com/google/flatbuffers/archive/v23.1.20.tar.gz"),
        build_file = "//third_party/flatbuffers:flatbuffers.BUILD",
        system_build_file = "//third_party/flatbuffers:BUILD.system",
        link_files = {
            "//third_party/flatbuffers:build_defs.bzl": "build_defs.bzl",
        },
    )

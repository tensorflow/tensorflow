"""Loads the Flatbuffers library, used by TF Lite."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "flatbuffers",
        strip_prefix = "flatbuffers-2.0.5",
        sha256 = "b01e97c988c429e164c5c7df9e87c80007ca87f593c0d73733ba536ddcbc8f98",
        urls = tf_mirror_urls("https://github.com/google/flatbuffers/archive/v2.0.5.tar.gz"),
        build_file = "//third_party/flatbuffers:flatbuffers.BUILD",
        system_build_file = "//third_party/flatbuffers:BUILD.system",
        link_files = {
            "//third_party/flatbuffers:build_defs.bzl": "build_defs.bzl",
        },
    )

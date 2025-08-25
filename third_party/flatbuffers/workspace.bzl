"""Loads the Flatbuffers library, used by TF Lite."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

_FLATBUFFERS_VERSION = "25.2.10"

# curl -L https://github.com/google/flatbuffers/archive/<_FLATBUFFERS_VERSION>.tar.gz | shasum -a 256
_FLATBUFFERS_SHA256 = "b9c2df49707c57a48fc0923d52b8c73beb72d675f9d44b2211e4569be40a7421"

def repo():
    tf_http_archive(
        name = "flatbuffers",
        strip_prefix = "flatbuffers-%s" % _FLATBUFFERS_VERSION,
        sha256 = _FLATBUFFERS_SHA256,
        urls = tf_mirror_urls("https://github.com/google/flatbuffers/archive/v%s.tar.gz" % _FLATBUFFERS_VERSION),
        build_file = "//third_party/flatbuffers:flatbuffers.BUILD",
        system_build_file = "//third_party/flatbuffers:BUILD.system",
        link_files = {
            "//third_party/flatbuffers:build_defs.bzl": "build_defs.bzl",
        },
    )

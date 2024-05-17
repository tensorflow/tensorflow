"""Loads the Flatbuffers library, used by TF Lite."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

# _FLATBUFFERS_GIT_COMMIT / _FLATBUFFERS_SHA256 were added due to an urgent change being made to
# Flatbuffers that needed to be updated in order for Flatbuffers/TfLite be compatible with Android
# API level >= 23. They can be removed next flatbuffers offical release / update.
_FLATBUFFERS_GIT_COMMIT = "e6463926479bd6b330cbcf673f7e917803fd5831"

# curl -L https://github.com/google/flatbuffers/archive/<_FLATBUFFERS_GIT_COMMIT>.tar.gz | shasum -a 256
_FLATBUFFERS_SHA256 = "c9c6b8653597ed7ee5c62243979010bd0f09b29a46be414505bc5b58a874bb17"

def repo():
    tf_http_archive(
        name = "flatbuffers",
        strip_prefix = "flatbuffers-%s" % _FLATBUFFERS_GIT_COMMIT,
        sha256 = _FLATBUFFERS_SHA256,
        urls = tf_mirror_urls("https://github.com/google/flatbuffers/archive/%s.tar.gz" % _FLATBUFFERS_GIT_COMMIT),
        build_file = "//third_party/flatbuffers:flatbuffers.BUILD",
        system_build_file = "//third_party/flatbuffers:BUILD.system",
        link_files = {
            "//third_party/flatbuffers:build_defs.bzl": "build_defs.bzl",
        },
    )

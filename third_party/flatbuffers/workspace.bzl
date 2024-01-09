"""Loads the Flatbuffers library, used by TF Lite."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

# _FLATBUFFERS_GIT_COMMIT / _FLATBUFFERS_SHA256 were added due to an urgent change being made to
# Flatbuffers that needed to be updated in order for Flatbuffers/TfLite be compatible with Android
# API level >= 23. They can be removed next flatbuffers offical release / update.
_FLATBUFFERS_GIT_COMMIT = "7d6d99c6befa635780a4e944d37ebfd58e68a108"

# curl -L https://github.com/google/flatbuffers/archive/<_FLATBUFFERS_GIT_COMMIT>.tar.gz | shasum -a 256
_FLATBUFFERS_SHA256 = "d27761f6b2fb1017ec00ed317a7b98cb7aed86b81d90528b498fb17ec13579a1"

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

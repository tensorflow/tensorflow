"""Loads the Flatbuffers library, used by TF Lite."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

# _FLATBUFFERS_GIT_COMMIT / _FLATBUFFERS_SHA256 were added due to an urgent change being made to
# Flatbuffers that needed to be updated in order for Flatbuffers/TfLite be compatible with Android
# API level >= 23. They can be removed next flatbuffers offical release / update.
_FLATBUFFERS_GIT_COMMIT = "6ff9e90e7e399f3977e99a315856b57c8afe5b4d"

# curl -L https://github.com/google/flatbuffers/archive/<_FLATBUFFERS_GIT_COMMIT>.tar.gz | shasum -a 256
_FLATBUFFERS_SHA256 = "f4b3dfed9f8f4f0fd9f857fe96a46199cb5745ddb458cad20caf6837230ea188"

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

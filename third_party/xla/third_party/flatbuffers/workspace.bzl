"""Loads the Flatbuffers library (WORKSPACE builds only; bzlmod builds use the BCR module)."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

_FLATBUFFERS_VERSION = "25.9.23"

# curl -L https://github.com/google/flatbuffers/archive/v<_FLATBUFFERS_VERSION>.tar.gz | shasum -a 256
_FLATBUFFERS_SHA256 = "9102253214dea6ae10c2ac966ea1ed2155d22202390b532d1dea64935c518ada"

def repo():
    tf_http_archive(
        name = "flatbuffers",
        strip_prefix = "flatbuffers-%s" % _FLATBUFFERS_VERSION,
        sha256 = _FLATBUFFERS_SHA256,
        urls = tf_mirror_urls("https://github.com/google/flatbuffers/archive/v%s.tar.gz" % _FLATBUFFERS_VERSION),
        build_file = "//third_party/flatbuffers:flatbuffers.BUILD",
    )

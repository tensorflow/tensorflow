"""Provides the repo macro to import google libprotobuf_mutator"""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Imports libprotobuf_mutator."""
    tf_http_archive(
        name = "com_google_libprotobuf_mutator",
        sha256 = "1ee3473a6b0274494fce599539605bb19305c0efadc62b58d645812132c31baa",
        strip_prefix = "libprotobuf-mutator-1.3",
        build_file = "//third_party/libprotobuf_mutator:libprotobuf_mutator.BUILD",
        urls = tf_mirror_urls("https://github.com/google/libprotobuf-mutator/archive/v1.3.tar.gz"),
    )

"""Provides the repo macro to import google libprotobuf_mutator"""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Imports libprotobuf_mutator."""
    tf_http_archive(
        name = "com_google_libprotobuf_mutator",
        sha256 = "792f250fb546bde8590e72d64311ea00a70c175fd77df6bb5e02328fa15fe28e",
        strip_prefix = "libprotobuf-mutator-1.0",
        build_file = "//third_party/libprotobuf_mutator:libprotobuf_mutator.BUILD",
        urls = tf_mirror_urls("https://github.com/google/libprotobuf-mutator/archive/v1.0.tar.gz"),
    )

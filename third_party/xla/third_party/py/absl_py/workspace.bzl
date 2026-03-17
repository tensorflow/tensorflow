"""Loads the abseil-py library, used by TF."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "absl_py",
        sha256 = "8a3d0830e4eb4f66c4fa907c06edf6ce1c719ced811a12e26d9d3162f8471758",
        strip_prefix = "abseil-py-2.1.0",
        urls = tf_mirror_urls("https://github.com/abseil/abseil-py/archive/refs/tags/v2.1.0.tar.gz"),
    )

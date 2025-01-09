"""Loads the nanobind library."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "nanobind",
        strip_prefix = "nanobind-2.4.0",
        sha256 = "bb35deaed7efac5029ed1e33880a415638352f757d49207a8e6013fefb6c49a7",
        urls = tf_mirror_urls("https://github.com/wjakob/nanobind/archive/refs/tags/v2.4.0.tar.gz"),
        build_file = "//third_party/nanobind:nanobind.BUILD",
    )

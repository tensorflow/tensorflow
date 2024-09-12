"""Loads the nanobind library."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "nanobind",
        strip_prefix = "nanobind-2.1.0",
        sha256 = "c37c53c60ada5fe1c956e24bd4b83af669a2309bf952bd251f36a7d2fa3bacf0",
        urls = tf_mirror_urls("https://github.com/wjakob/nanobind/archive/refs/tags/v2.1.0.tar.gz"),
        build_file = "//third_party/nanobind:nanobind.BUILD",
    )

"""Loads the nanobind library."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "nanobind",
        strip_prefix = "nanobind-2.7.0",
        sha256 = "6c8c6bf0435b9d8da9312801686affcf34b6dbba142db60feec8d8e220830499",
        urls = tf_mirror_urls("https://github.com/wjakob/nanobind/archive/refs/tags/v2.7.0.tar.gz"),
        build_file = "//third_party/nanobind:nanobind.BUILD",
    )

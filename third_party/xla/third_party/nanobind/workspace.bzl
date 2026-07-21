"""Loads the nanobind library."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "nanobind",
        strip_prefix = "nanobind-e2dc00f7a34f935c6cf91948776d59c4709e9fe6",
        sha256 = "99fea0ea1c61b94a02811f7ad4915e70145b8acdb4b65bb67a4e56981d1f7d32",
        urls = tf_mirror_urls("https://github.com/wjakob/nanobind/archive/e2dc00f7a34f935c6cf91948776d59c4709e9fe6.tar.gz"),
        build_file = "//third_party/nanobind:nanobind.BUILD",
    )

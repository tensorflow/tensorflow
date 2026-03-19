"""Loads the nanobind library."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "nanobind",
        strip_prefix = "nanobind-54d509920c8e775710de479ba7ec6c7198979038",
        sha256 = "1eff8703b9260b4371fa03bd6f9d7e8a715dc0004809f26c5204d610eedee056",
        urls = tf_mirror_urls("https://github.com/wjakob/nanobind/archive/54d509920c8e775710de479ba7ec6c7198979038.tar.gz"),
        build_file = "//third_party/nanobind:nanobind.BUILD",
    )

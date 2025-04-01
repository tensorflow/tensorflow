"""Loads the nanobind library."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "nanobind",
        strip_prefix = "nanobind-b4b933111fa61815f3f5b509fde80c24f029ac26",
        sha256 = "d1d8575f2bf76cc2ca357ac5521daa2f1bcf5397231d856f4ce66ba0670ac928",
        urls = tf_mirror_urls("https://github.com/wjakob/nanobind/archive/b4b933111fa61815f3f5b509fde80c24f029ac26.tar.gz"),
        build_file = "//third_party/nanobind:nanobind.BUILD",
    )

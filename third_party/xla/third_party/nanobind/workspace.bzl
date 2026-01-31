"""Loads the nanobind library."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "nanobind",
        strip_prefix = "nanobind-2.11.0",
        sha256 = "62ba05e5f720c76c510d6ab2a77f8ccc17a76c5cea951bea47355a7dfa460449",
        urls = tf_mirror_urls("https://github.com/wjakob/nanobind/archive/refs/tags/v2.11.0.tar.gz"),
        build_file = "//third_party/nanobind:nanobind.BUILD",
    )

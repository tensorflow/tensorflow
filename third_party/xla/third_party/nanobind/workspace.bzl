"""Loads the nanobind library."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "nanobind",
        strip_prefix = "nanobind-2.12.0",
        sha256 = "01f1f0cd0398743c18f33d07ae36ad410bd7f4a1e90683b508504de897d6e629",
        urls = tf_mirror_urls("https://github.com/wjakob/nanobind/archive/refs/tags/v2.12.0.tar.gz"),
        build_file = "//third_party/nanobind:nanobind.BUILD",
    )

"""Loads the nanobind library."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "nanobind",
        strip_prefix = "nanobind-2.9.2",
        sha256 = "8ce3667dce3e64fc06bfb9b778b6f48731482362fb89a43da156632266cd5a90",
        urls = tf_mirror_urls("https://github.com/wjakob/nanobind/archive/refs/tags/v2.9.2.tar.gz"),
        build_file = "//third_party/nanobind:nanobind.BUILD",
    )

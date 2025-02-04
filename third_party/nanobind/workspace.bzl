"""Loads the nanobind library."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "nanobind",
        strip_prefix = "nanobind-d79309197caaad83cda05df533136865d294f01e",
        sha256 = "598b116f36dbdf9738bb269cc1551ae073715fb3d69f07ca0dd01e6de0ddf4b0",
        urls = tf_mirror_urls("https://github.com/wjakob/nanobind/archive/d79309197caaad83cda05df533136865d294f01e.tar.gz"),
        build_file = "//third_party/nanobind:nanobind.BUILD",
    )

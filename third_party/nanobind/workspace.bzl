"""Loads the nanobind library."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "nanobind",
        strip_prefix = "nanobind-2.2.0",
        sha256 = "bfbfc7e5759f1669e4ddb48752b1ddc5647d1430e94614d6f8626df1d508e65a",
        urls = tf_mirror_urls("https://github.com/wjakob/nanobind/archive/refs/tags/v2.2.0.tar.gz"),
        build_file = "//third_party/nanobind:nanobind.BUILD",
    )

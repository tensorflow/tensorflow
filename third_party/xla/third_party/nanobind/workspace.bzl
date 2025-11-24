"""Loads the nanobind library."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "nanobind",
        strip_prefix = "nanobind-e507b118927bc3a12446d0ca235e1baaf343932e",
        sha256 = "95004c4cd1f3e7417b71ff25be9cfba7e8ad79e570248e377815bc980c8b3c73",
        urls = tf_mirror_urls("https://github.com/wjakob/nanobind/archive/e507b118927bc3a12446d0ca235e1baaf343932e.tar.gz"),
        build_file = "//third_party/nanobind:nanobind.BUILD",
    )

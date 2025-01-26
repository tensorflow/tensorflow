"""Loads the nanobind library."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "nanobind",
        strip_prefix = "nanobind-cee104db8606797a63752d2904b2f2795014a125",
        sha256 = "d5dec3690c0a11b1ca48021ff34238886da7938b7bbbd5c0e946dcef6e6b7e25",
        urls = tf_mirror_urls("https://github.com/wjakob/nanobind/archive/cee104db8606797a63752d2904b2f2795014a125.tar.gz"),
        build_file = "//third_party/nanobind:nanobind.BUILD",
    )

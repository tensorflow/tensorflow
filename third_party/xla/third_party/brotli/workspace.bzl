"""Provides the repo macro to import brotli"""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "org_brotli",
        sha256 = "e720a6ca29428b803f4ad165371771f5398faba397edf6778837a18599ea13ff",
        strip_prefix = "brotli-1.1.0",
        urls = tf_mirror_urls("https://github.com/google/brotli/archive/refs/tags/v1.1.0.tar.gz"),
    )

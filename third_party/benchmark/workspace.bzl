"""Provides the repo macro to import google benchmark"""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Imports benchmark."""
    BM_COMMIT = "617bfb06b7df09431ec1449c7810f02e459a9fd5"
    BM_SHA256 = "8ec1c24fdaa67caf923b62211383ae35f6d1500d98401d922abf56de3d81d992"
    tf_http_archive(
        name = "com_google_benchmark",
        sha256 = BM_SHA256,
        strip_prefix = "benchmark-{commit}".format(commit = BM_COMMIT),
        urls = tf_mirror_urls("https://github.com/google/benchmark/archive/{commit}.tar.gz".format(commit = BM_COMMIT)),
    )

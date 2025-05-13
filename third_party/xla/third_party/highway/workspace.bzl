"""Loads the highway library."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "highway",
        urls = tf_mirror_urls("https://github.com/google/highway/archive/8b432f3f3eda0a1342fc64c0762b5390f1a2ca0d.tar.gz"),
        sha256 = "95ead857f3658edb157015f640ee704cab3ba343aeff9fcd06c5091317cd2480",
        strip_prefix = "highway-8b432f3f3eda0a1342fc64c0762b5390f1a2ca0d",
    )

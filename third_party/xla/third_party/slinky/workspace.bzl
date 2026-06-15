"""slinky is a lightweight runtime for semi-automatical optimization of data flow pipelines for locality."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "slinky",
        sha256 = "35649e12cefd1d0cb02f0e5dea9b012cf8ebd6db0315065b5a43fccb3009a552",
        strip_prefix = "slinky-bba574d26d63dccd9d47e95b0e039b38cd0fb0de",
        urls = tf_mirror_urls("https://github.com/dsharlet/slinky/archive/bba574d26d63dccd9d47e95b0e039b38cd0fb0de.zip"),
    )

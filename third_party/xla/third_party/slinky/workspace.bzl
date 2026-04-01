"""slinky is a lightweight runtime for semi-automatical optimization of data flow pipelines for locality."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "slinky",
        sha256 = "28d311d7aa3c7f432e3fc14a09f11bff363a1611896c198d1c4a41ed5aab0d48",
        strip_prefix = "slinky-4de79eb693dfa2791fa469586c8052287cb3110d",
        urls = tf_mirror_urls("https://github.com/dsharlet/slinky/archive/4de79eb693dfa2791fa469586c8052287cb3110d.zip"),
    )

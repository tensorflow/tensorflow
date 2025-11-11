"""slinky is a lightweight runtime for semi-automatical optimization of data flow pipelines for locality."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "slinky",
        sha256 = "d0672d8abae3a0cebad6245ed7d8838b101b343daeaf2ed0dbd3e0769ac4f386",
        strip_prefix = "slinky-00f549edc2e9d3df74abc0ff527270c58b5dda6c",
        urls = tf_mirror_urls("https://github.com/dsharlet/slinky/archive/00f549edc2e9d3df74abc0ff527270c58b5dda6c.zip"),
    )

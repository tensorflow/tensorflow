"""slinky is a lightweight runtime for semi-automatical optimization of data flow pipelines for locality."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "slinky",
        sha256 = "4c0b19e58706c44336c573a0ba9fdf0e412b23670cac6a3df95525a0909a0360",
        strip_prefix = "slinky-2afe84b39f0d097ecd70fc44a9e0e39782cee6a3",
        urls = tf_mirror_urls("https://github.com/dsharlet/slinky/archive/2afe84b39f0d097ecd70fc44a9e0e39782cee6a3.zip"),
    )

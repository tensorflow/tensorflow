"""slinky is a lightweight runtime for semi-automatical optimization of data flow pipelines for locality."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "slinky",
        sha256 = "cd84dd2ff1956480fac6e50f477441d74e0ad08e34e863ffad2a1d3adf043dc2",
        strip_prefix = "slinky-eb004cb32169f87a4d95ee73af4eb9cf3df42868",
        urls = tf_mirror_urls("https://github.com/dsharlet/slinky/archive/eb004cb32169f87a4d95ee73af4eb9cf3df42868.zip"),
    )

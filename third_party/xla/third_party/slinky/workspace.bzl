"""slinky is a lightweight runtime for semi-automatical optimization of data flow pipelines for locality."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "slinky",
        sha256 = "d72196a2feda99ad3fa8f724ee8ab6b1fe2883ad4cb12d2a7169b90b833e0301",
        strip_prefix = "slinky-cb1f285b76c53b6e8b4075d25127140e5b985f8e",
        urls = tf_mirror_urls("https://github.com/dsharlet/slinky/archive/cb1f285b76c53b6e8b4075d25127140e5b985f8e.zip"),
    )

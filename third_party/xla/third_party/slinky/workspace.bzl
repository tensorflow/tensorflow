"""slinky is a lightweight runtime for semi-automatical optimization of data flow pipelines for locality."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "slinky",
        sha256 = "be1da0df6555924ede8ddfc43495dd61b1eda3654263063931424bced1de3bc9",
        strip_prefix = "slinky-1032be67d7033d736eb6afbb9b51865baa5c77ae",
        urls = tf_mirror_urls("https://github.com/dsharlet/slinky/archive/1032be67d7033d736eb6afbb9b51865baa5c77ae.zip"),
    )

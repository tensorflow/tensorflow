"""slinky is a lightweight runtime for semi-automatical optimization of data flow pipelines for locality."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "slinky",
        sha256 = "659253670a5c7983bcebed6f2190b48024465a3beedcfcd7239acc0ae73778fb",
        strip_prefix = "slinky-4a497c5bd9e52588a5579d4c7ab3a6983e5a8b1d",
        urls = tf_mirror_urls("https://github.com/dsharlet/slinky/archive/4a497c5bd9e52588a5579d4c7ab3a6983e5a8b1d.zip"),
    )

"""slinky is a lightweight runtime for semi-automatical optimization of data flow pipelines for locality."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "slinky",
        sha256 = "eee999088e6105043663abf22a00268b22013151166109fafb15635da81f2de6",
        strip_prefix = "slinky-8eda4a050105a7148d3d6cb77960077790d69b01",
        urls = tf_mirror_urls("https://github.com/dsharlet/slinky/archive/8eda4a050105a7148d3d6cb77960077790d69b01.zip"),
    )

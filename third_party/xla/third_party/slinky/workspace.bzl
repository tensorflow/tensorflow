"""slinky is a lightweight runtime for semi-automatical optimization of data flow pipelines for locality."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "slinky",
        sha256 = "e4e72c7b45fc3b964a76fe3d2f7af7f3c135eaba88112e08510143dbde1c9b7e",
        strip_prefix = "slinky-dad55945ad3c70d3268ffc043078469db810cd03",
        urls = tf_mirror_urls("https://github.com/dsharlet/slinky/archive/dad55945ad3c70d3268ffc043078469db810cd03.zip"),
    )

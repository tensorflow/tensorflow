"""slinky is a lightweight runtime for semi-automatical optimization of data flow pipelines for locality."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "slinky",
        sha256 = "f9e718f65bcf2710450e00b0ed383a1025bc9a8bf3abfda85e49587f9f34929d",
        strip_prefix = "slinky-36852ece52b3101a5c56b741c20866988428ae21",
        urls = tf_mirror_urls("https://github.com/dsharlet/slinky/archive/36852ece52b3101a5c56b741c20866988428ae21.zip"),
    )

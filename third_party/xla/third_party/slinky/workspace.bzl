"""slinky is a lightweight runtime for semi-automatical optimization of data flow pipelines for locality."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "slinky",
        sha256 = "1cea5f5ee913da8a7cb703653c14c9b0453ce61216512560abc4414b5504f4fc",
        strip_prefix = "slinky-ed97d801653a6680d4b558614f323f2a01d33413",
        urls = tf_mirror_urls("https://github.com/dsharlet/slinky/archive/ed97d801653a6680d4b558614f323f2a01d33413.zip"),
    )

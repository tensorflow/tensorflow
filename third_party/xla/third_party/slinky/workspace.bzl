"""slinky is a lightweight runtime for semi-automatical optimization of data flow pipelines for locality."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "slinky",
        sha256 = "e150493794cebe3407523bad5fb63e844d9c4f0313dd4dfb49ebcad29d6172de",
        strip_prefix = "slinky-8b06c0f25578c34b163f430b92f7ce923ed96ff4",
        urls = tf_mirror_urls("https://github.com/dsharlet/slinky/archive/8b06c0f25578c34b163f430b92f7ce923ed96ff4.zip"),
    )

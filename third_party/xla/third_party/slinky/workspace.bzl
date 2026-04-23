"""slinky is a lightweight runtime for semi-automatical optimization of data flow pipelines for locality."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "slinky",
        sha256 = "8edbc4b16ea3c582148eaace6f120fba1be5571496f48dd132605fc53f9d8e6f",
        strip_prefix = "slinky-92c3492bece13d1247025ceff33275ad1b725a8e",
        urls = tf_mirror_urls("https://github.com/dsharlet/slinky/archive/92c3492bece13d1247025ceff33275ad1b725a8e.zip"),
    )

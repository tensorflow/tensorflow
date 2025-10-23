"""slinky is a lightweight runtime for semi-automatical optimization of data flow pipelines for locality."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "slinky",
        sha256 = "e30625e2a3071ed674d48436bd671547f29825953ea7349c066c21322587ff6e",
        strip_prefix = "slinky-e9c16e794b1a0a5f7bcc86565108a325f98c1422",
        urls = tf_mirror_urls("https://github.com/dsharlet/slinky/archive/e9c16e794b1a0a5f7bcc86565108a325f98c1422.zip"),
    )

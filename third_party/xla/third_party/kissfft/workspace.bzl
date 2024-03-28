"""Loads the kissfft library, used by TF Lite."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "kissfft",
        strip_prefix = "kissfft-131.1.0",
        sha256 = "76c1aac87ddb7258f34b08a13f0eebf9e53afa299857568346aa5c82bcafaf1a",
        urls = tf_mirror_urls("https://github.com/mborgerding/kissfft/archive/refs/tags/131.1.0.tar.gz"),
        build_file = "//third_party/kissfft:kissfft.BUILD",
    )

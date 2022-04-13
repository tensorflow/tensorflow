"""Loads the kissfft library, used by TF Lite."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "kissfft",
        strip_prefix = "kissfft-36dbc057604f00aacfc0288ddad57e3b21cfc1b8",
        sha256 = "42b7ef406d5aa2d57a7b3b56fc44e8ad3011581692458a69958a911071efdcf2",
        urls = tf_mirror_urls("https://github.com/mborgerding/kissfft/archive/36dbc057604f00aacfc0288ddad57e3b21cfc1b8.tar.gz"),
        build_file = "//third_party/kissfft:kissfft.BUILD",
    )

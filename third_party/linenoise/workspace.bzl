"""Loads the linenoise library, used by TF."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "linenoise",
        build_file = "//third_party/linenoise:linenoise.BUILD",
        sha256 = "b35a74dbc9cd2fef9e4d56222761d61daf7e551510e6cd1a86f0789b548d074e",
        strip_prefix = "linenoise-4ce393a66b10903a0ef52edf9775ed526a17395f",
        urls = tf_mirror_urls("https://github.com/antirez/linenoise/archive/4ce393a66b10903a0ef52edf9775ed526a17395f.tar.gz"),
    )

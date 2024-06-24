"""Loads the robin_map library."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "robin_map",
        strip_prefix = "robin-map-1.3.0",
        sha256 = "a8424ad3b0affd4c57ed26f0f3d8a29604f0e1f2ef2089f497f614b1c94c7236",
        urls = tf_mirror_urls("https://github.com/Tessil/robin-map/archive/refs/tags/v1.3.0.tar.gz"),
        build_file = "//third_party/robin_map:robin_map.BUILD",
    )

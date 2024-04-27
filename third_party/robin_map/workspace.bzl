"""Loads the robin_map library."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "robin_map",
        strip_prefix = "robin-map-1.2.1",
        sha256 = "2b54d2c1de2f73bea5c51d5dcbd64813a08caf1bfddcfdeee40ab74e9599e8e3",
        urls = tf_mirror_urls("https://github.com/Tessil/robin-map/archive/refs/tags/v1.2.1.tar.gz"),
        build_file = "//third_party/robin_map:robin_map.BUILD",
    )

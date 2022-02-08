"""Loads the clog library, used by cpuinfo and XNNPACK."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "clog",
        strip_prefix = "cpuinfo-d5e37adf1406cf899d7d9ec1d317c47506ccb970",
        sha256 = "3f2dc1970f397a0e59db72f9fca6ff144b216895c1d606f6c94a507c1e53a025",
        urls = tf_mirror_urls("https://github.com/pytorch/cpuinfo/archive/d5e37adf1406cf899d7d9ec1d317c47506ccb970.tar.gz"),
        build_file = "//third_party/clog:clog.BUILD",
    )

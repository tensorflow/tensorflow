"""oneAPI Data Parallel C++ Library (oneDPL)"""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "onedpl",
        build_file = "//third_party/onedpl:onedpl.BUILD",
        sha256 = "1e40549d300265aac5459e6e52b5aa5ae9073586067c4a4a38a21c3511ca0373",
        strip_prefix = "oneDPL-oneDPL-release-2022.14.0",
        urls = tf_mirror_urls("https://github.com/uxlfoundation/oneDPL/archive/refs/tags/oneDPL-release-2022.14.0.tar.gz"),
    )

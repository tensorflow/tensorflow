"""oneAPI Data Parallel C++ Library (oneDPL)"""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "onedpl",
        build_file = "//third_party/onedpl:onedpl.BUILD",
        sha256 = "d0524a3c896616335aef42d244cb331b31fec14cf7389cdd0e89c2e5b28ecf77",
        strip_prefix = "oneDPL-oneDPL-release-2022.13.0",
        urls = tf_mirror_urls("https://github.com/uxlfoundation/oneDPL/archive/refs/tags/oneDPL-release-2022.13.0.tar.gz"),
    )

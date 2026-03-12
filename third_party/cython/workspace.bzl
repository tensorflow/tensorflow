"""Loads the cython library, used by TF."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "cython",
        build_file = "@xla//third_party:cython.BUILD",
        sha256 = "da72f94262c8948e04784c3e6b2d14417643703af6b7bd27d6c96ae7f02835f1",
        strip_prefix = "cython-3.1.2",
        system_build_file = "//third_party/systemlibs:cython.BUILD",
        urls = tf_mirror_urls("https://github.com/cython/cython/archive/3.1.2.tar.gz"),
    )

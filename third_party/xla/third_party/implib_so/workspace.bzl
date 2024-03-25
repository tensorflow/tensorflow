"""Implib.so is a simple equivalent of Windows DLL import libraries for POSIX
shared libraries."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "implib_so",
        strip_prefix = "Implib.so-5fb84c2a750434b9df1da67d67b749eb929598f1",
        sha256 = "10de0a616df24849f2a883747784c115f209708960e44556f5ce384de6f103e8",
        urls = tf_mirror_urls("https://github.com/yugr/Implib.so/archive/5fb84c2a750434b9df1da67d67b749eb929598f1.tar.gz"),
        build_file = "//third_party/implib_so:implib_so.BUILD",
    )

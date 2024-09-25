"""Implib.so is a simple equivalent of Windows DLL import libraries for POSIX
shared libraries."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "implib_so",
        strip_prefix = "Implib.so-2cce6cab8ff2c15f9da858ea0b68646a8d62aef2",
        sha256 = "4ef3089969d57a5b60bb41b8212c478eaa15c56941f86d4bf5e7f98a3afd24e8",
        urls = tf_mirror_urls("https://github.com/yugr/Implib.so/archive/2cce6cab8ff2c15f9da858ea0b68646a8d62aef2.tar.gz"),
        build_file = "//third_party/implib_so:implib_so.BUILD",
    )

"""xxd is a tool that converts a binary file into a C/C++ header file containing an array of bytes."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "xxd",
        sha256 = "a5cdcfcfeb13dc4deddcba461d40234dbf47e61941cb7170c9ebe147357bb62d",
        strip_prefix = "vim-9.1.0917/src/xxd",
        urls = tf_mirror_urls("https://github.com/vim/vim/archive/refs/tags/v9.1.0917.tar.gz"),
        build_file = "//third_party/xxd:xxd.BUILD",
    )

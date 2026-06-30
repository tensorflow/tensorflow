"""Loads the nanobind library."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "nanobind",
        strip_prefix = "nanobind-30f12ae6650ecec86042053d522d9af585f269b0",
        sha256 = "8948a72b93ddf3846e1fb894ddf825794138c65f23e76f1138f5e20a73cf7b10",
        urls = tf_mirror_urls("https://github.com/wjakob/nanobind/archive/30f12ae6650ecec86042053d522d9af585f269b0.tar.gz"),
        build_file = "//third_party/nanobind:nanobind.BUILD",
    )

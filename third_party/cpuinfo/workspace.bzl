"""Loads the cpuinfo library, used by XNNPACK."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "cpuinfo",
        strip_prefix = "cpuinfo-b40bae27785787b6dd70788986fd96434cf90ae2",
        sha256 = "5794c7b37facc590018eddffec934c60aeb71165b59a375babe4f7be7f04723f",
        urls = tf_mirror_urls("https://github.com/pytorch/cpuinfo/archive/b40bae27785787b6dd70788986fd96434cf90ae2.tar.gz"),
        build_file = "//third_party/cpuinfo:cpuinfo.BUILD",
    )

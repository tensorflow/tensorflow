"""Loads the cpuinfo library, used by XNNPACK."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "cpuinfo",
        strip_prefix = "cpuinfo-9fa621933fc6080b96fa0f037cdc7cd2c69ab272",
        sha256 = "810708948128be2da882a5a3ca61eb6db40186bac9180d205a7ece43597b5fc3",
        urls = tf_mirror_urls("https://github.com/pytorch/cpuinfo/archive/9fa621933fc6080b96fa0f037cdc7cd2c69ab272.tar.gz"),
        build_file = "//third_party/cpuinfo:cpuinfo.BUILD",
    )

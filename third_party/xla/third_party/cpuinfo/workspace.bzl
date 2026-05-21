"""cpuinfo is a library to detect essential CPU features."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "cpuinfo",
        sha256 = "5d7f00693e97bd7525753de94be63f99b0490ae6855df168f5a6b2cfc452e49e",
        strip_prefix = "cpuinfo-3c8b1533ac03dd6531ab6e7b9245d488f13a82a5",
        urls = tf_mirror_urls("https://github.com/pytorch/cpuinfo/archive/3c8b1533ac03dd6531ab6e7b9245d488f13a82a5.zip"),
    )

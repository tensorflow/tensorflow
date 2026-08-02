"""cpuinfo is a library to detect essential CPU features."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "cpuinfo",
        sha256 = "9213f6f81784eb8679f0621ad1c20eac711e063cb9c7712738720609cbdf1c33",
        strip_prefix = "cpuinfo-ea6b9f1bb6e1001d8b21574d5bc78ddef62e499d",
        urls = tf_mirror_urls("https://github.com/pytorch/cpuinfo/archive/ea6b9f1bb6e1001d8b21574d5bc78ddef62e499d.zip"),
    )

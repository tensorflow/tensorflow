"""Loads the cpuinfo library, used by XNNPACK."""

load("//third_party:repo.bzl", "third_party_http_archive")

def repo():
    third_party_http_archive(
        name = "cpuinfo",
        strip_prefix = "cpuinfo-5cefcd6293e6881754c2c53f99e95b159d2d8aa5",
        sha256 = "8ea076bcc4ff73cdff520ece01b776d2a778ced60956f5eb88697a78e22c389d",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/pytorch/cpuinfo/archive/5cefcd6293e6881754c2c53f99e95b159d2d8aa5.zip",
            "https://github.com/pytorch/cpuinfo/archive/5cefcd6293e6881754c2c53f99e95b159d2d8aa5.zip",
        ],
        build_file = "//third_party/cpuinfo:BUILD.bazel",
    )

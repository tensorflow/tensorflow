"""Loads the cpuinfo library, used by XNNPACK."""

load("//third_party:repo.bzl", "third_party_http_archive")

def repo():
    third_party_http_archive(
        name = "cpuinfo",
        strip_prefix = "cpuinfo-e39a5790059b6b8274ed91f7b5b5b13641dff267",
        sha256 = "e5caa8b7c58f1623eed88f4d5147e3753ff19cde821526bc9aa551b004f751fe",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/pytorch/cpuinfo/archive/e39a5790059b6b8274ed91f7b5b5b13641dff267.tar.gz",
            "https://github.com/pytorch/cpuinfo/archive/e39a5790059b6b8274ed91f7b5b5b13641dff267.tar.gz",
        ],
        build_file = "//third_party/cpuinfo:BUILD.bazel",
    )

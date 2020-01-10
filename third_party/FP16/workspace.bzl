"""Loads the FP16 library, used by TF Lite."""

load("//third_party:repo.bzl", "third_party_http_archive")

def repo():
    third_party_http_archive(
        name = "FP16",
        strip_prefix = "FP16-febbb1c163726b5db24bed55cc9dc42529068997",
        sha256 = "3e71681e0a67cd28552aa0bbb78ec6a6bd238216df15336dc1326280f7958de2",
        urls = [
            "https://mirror.bazel.build/github.com/Maratyszcza/FP16/archive/febbb1c163726b5db24bed55cc9dc42529068997.tar.gz",
            "https://github.com/Maratyszcza/FP16/archive/febbb1c163726b5db24bed55cc9dc42529068997.tar.gz",
        ],
        build_file = "//third_party/FP16:BUILD.bazel",
    )
